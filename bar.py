import os
from io import BytesIO
import openai
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import base64
from typing import Dict, Optional, Union, Any
from openai import OpenAI
from sklearn.cluster import MiniBatchKMeans
from PIL import Image

class GraphGenerator:
    def __init__(self):
        # 定义安全执行环境
        self.safe_modules = {
            "plt": plt,
            "np": np,
            "math": __import__("math")
        }
        # 初始化gpt
        self.client = OpenAI()
        # 读取当前的计数器值
        self.data_save_folder = "generated_data_bar"  # 保存图片的文件夹
        if not os.path.exists(self.data_save_folder):
            os.makedirs(self.data_save_folder)  # 如果文件夹不存在，则创建

        # 读取当前的计数器值，若不存在则初始化为 1
        self.counter_file_path = os.path.join(self.data_save_folder, "counter.txt")
        if os.path.exists(self.counter_file_path):
            with open(self.counter_file_path, "r") as f:
                self.data_counter = int(f.read())
        else:
            self.data_counter = 1  # 若文件不存在，则初始化为 1

    def call_gpt_and_generate(
            self,
            initial_instruction: str,
            requirement: str,
            # sample_answer: str,
            student_answer: str,
            image_path: Optional[str] = None,
            model: str = "gpt-4o",
            output_format: str = "json",  # 支持json/cod
            chartvlm_data=None) -> Union[Optional[Dict[str, str]], Any]:
        """
        主执行方法：发送请求到DeepSeek API并处理响应
        """
        try:
            # 构建系统指令
            system_content = """You are an IELTS Academic Writing Task 1 expert. Your tasks:
1. Analyze student answers to extract numerical data AND trend information
2. For trend descriptions (e.g., "increased steadily", "sharp decline"):
   - Identify start/end points when explicitly mentioned
   - Estimate intermediate values using linear interpolation when only trend is described
   - Estimated values must be strings with "[est]" prefix (e.g., "[est] 85")
3. Missing data handling:
   - If a data point is completely missing and no trend is given, set the value to null.
   - If a trend is given but intermediate values are missing, you must estimate them.
4.Color extraction from image descriptions:
   - Identify colors associated with each data series in legends
   - Convert colors to HEX format (e.g., "#FF0000" for red)
   - If no color info available, return empty list
5. Never include any executable code.
6. Output JSON format:
{
  "title": "...",
  "categories": [...],
  "x_label": "...",
  "y_label": "...",
  "series": [
    {
      "label": "...",
      "values": [...],
      "trend_info": [
        {"type": "increase/decrease/stable/fluctuate", "strength": "slight/moderate/dramatic", "time_range": ["start", "end"]},
        ...
      ]
    }
  ],
  "chart_type": "bar/line",
  "style": {
    "orientation": "horizontal/vertical",
    "color_palette": [],
    "estimated_values": [indexes]
  }
}

"""
            # 构建消息队列
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Task Context: {initial_instruction}"},
                {"role": "user", "content": f"Official Requirement: {requirement}"},
                # {"role": "user", "content": f"Sample Answer: {sample_answer}"},
                {"role": "user", "content": f"""
                    ## Student Answer Analysis Task
                    * Primary Data Source: STUDENT ANSWER
                    * Conflict Resolution: Prioritize student's version when conflicting with sample
                    * Special Markers: Use [?] for unconfirmed data
                    [BEGIN STUDENT ANSWER]
                    {student_answer}
                    [END STUDENT ANSWER]
                    """}
            ]

            if image_path:
                # 提取图像的颜色调色板并作为用户提示
                # palette = self.extract_color_palette(image_path)
                # if palette:
                #     palette_str = ", ".join(f'"{c}"' for c in palette)
                #     messages.append({"role": "user",
                #                      "content": f"The original graph uses the following color palette: [{palette_str}]"})
                # 添加编码后的图像
                messages.append(self._encode_image(image_path))

            # 调用 API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1500,
            )

            if not response or not hasattr(response, 'choices') or not response.choices:
                print("⚠️ DeepSeek API返回空响应")
                return {"error": "Empty response from DeepSeek API"}

            content_text = response.choices[0].message.content
            print(f"📝 DeepSeek API返回: {content_text[:100]}...")  # 打印前100个字符

            # 保存原始响应内容以便调试
            with open("debug.txt", "w", encoding="utf-8") as f:
                f.write(content_text)
            print("📝 完整响应已保存到 debug.txt")

            if output_format == "json":
                try:
                    # 解析JSON内容
                    content = self._process_response(content_text, output_format)

                    # 验证内容是否包含必要字段
                    if not content.get("series") or not content.get("categories"):
                        print("⚠️ DeepSeek返回的JSON不包含必要的字段")
                        return {"error": "Invalid JSON structure - missing required fields"}

                    # 补全趋势数据
                    content = self._complete_trend_data(content)
                    print("📊 deepseek JSON 处理完成，包含趋势数据")

                    # 仅当 chartvlm 数据存在时才进行更新
                    if chartvlm_data:
                        updated_content = self.preprocess_and_update(content, chartvlm_data)
                        return updated_content
                    else:
                        return content
                except Exception as e:
                    print(f"❌ JSON处理错误: {str(e)}")
                    return {"error": f"JSON processing error: {str(e)}"}

            return {"error": "Unsupported output format"}

        except Exception as e:
            print(f"❌ API Error: {str(e)}")
            return {"error": str(e)}

    # def _encode_image(self, image_path: str) -> Dict:
    #     """
    #     图片编码方法：将图片转换为base64字符串并以纯文本形式传递
    #     """
        # 打开图片并压缩尺寸
        # with Image.open(image_path) as img:
        #     img.thumbnail((1000, 1000))  # 限制最大宽高，保持原图比例
        #
        #     # 压缩并转为JPEG格式保存到内存
        #     buffer = BytesIO()
        #     img.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=False)
        #
        #     # 编码为 base64 字符串
        #     encoded_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        #
        #     # 拼接为最终内容
        #     content_str = f"Reference image provided: data:image/jpeg;base64,{encoded_data}"
        #     return {
        #         "role": "user",
        #         "content": content_str
        #     }


    def _encode_image(self, image_path: str) -> Dict:
        """
        图片编码方法：将图片转换为 base64，并以结构化消息的形式传递，
        符合 OpenAI 图像消息格式。返回带有文本提示和 image_url 的消息片段。
        """
        # 打开图片并压缩尺寸
        with Image.open(image_path) as img:
            img.thumbnail((1000, 1000))  # 限制最大宽高，保持原图比例

            # 转为 JPEG 并压缩保存到内存
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=False)

            # 编码为 base64 字符串
            encoded_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{encoded_data}"

        # 返回符合 OpenAI ChatCompletion 接收图片的结构化消息
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Reference image provided (请仔细识别图例颜色及对应类别)："
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri
                    }
                }
            ]
        }

    def extract_color_palette(self, image_path: str, max_colors: int = 5) -> list:
        def rgb_to_hsv(rgb):
            # 将RGB转为HSV色彩空间
            rgb = np.array(rgb) / 255.0
            maxc = np.max(rgb)
            minc = np.min(rgb)
            v = maxc
            if minc == maxc:
                return (0.0, 0.0, v)
            s = (maxc - minc) / maxc
            rc = (maxc - rgb[0]) / (maxc - minc)
            gc = (maxc - rgb[1]) / (maxc - minc)
            bc = (maxc - rgb[2]) / (maxc - minc)
            h = 0.0
            if rgb[0] == maxc:
                h = bc - gc
            elif rgb[1] == maxc:
                h = 2.0 + rc - bc
            else:
                h = 4.0 + gc - rc
            h = (h / 6.0) % 1.0
            return (h, s, v)

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.width * img.height > 300000:
                img = img.resize((300, 300))  # 保持更多细节

            data = np.array(img)
            pixels = data.reshape(-1, 3)

            # 多维度过滤条件
            filtered_pixels = []
            for pixel in pixels:
                r, g, b = [int(x) for x in pixel]
                h, s, v = rgb_to_hsv(pixel)

                cond1 = v < 0.85  # 排除过亮颜色
                cond2 = s > 0.25  # 保证饱和度
                cond3 = not (abs(r - g) < 25 and abs(g - b) < 25 and abs(r - b) < 25)  # 排除近似灰色
                cond4 = (max(r, g, b) - min(r, g, b)) > 40

            if cond1 and cond2 and cond3 and cond4:
                filtered_pixels.append(pixel)

            if not filtered_pixels:
                return ["#1E90FF", "#FFA500"]  # 默认备用颜色

            pixels = np.array(filtered_pixels, dtype=np.float32)

            # 动态调整聚类数量
            actual_colors = min(max_colors, len(np.unique(pixels, axis=0)))
            if actual_colors < 2:
                return ['#%02x%02x%02x' % tuple(pixels[0])]

            # K-means聚类改进版

            kmeans = MiniBatchKMeans(
                n_clusters=actual_colors,
                random_state=42,
                batch_size=1024,
                n_init=3
            )
            kmeans.fit(pixels)

            # 按聚类大小排序，优先选择主要颜色
            unique, counts = np.unique(kmeans.labels_, return_counts=True)
            sorted_colors = kmeans.cluster_centers_[np.argsort(-counts)]

            # 转换为十六进制
            palette = []
            for color in sorted_colors:
                hex_color = '#%02x%02x%02x' % tuple(color.astype(int))

                # 确保颜色差异度 (与已选颜色对比)
                if not palette or all(
                        self._color_distance(hex_color, c) > 30
                        for c in palette
                ):
                    palette.append(hex_color)
                    if len(palette) >= max_colors:
                        break

            return palette

    @staticmethod
    def _color_distance(c1: str, c2: str) -> float:
        """计算两个颜色之间的欧氏距离"""

        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        r1, g1, b1 = hex_to_rgb(c1)
        r2, g2, b2 = hex_to_rgb(c2)
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5

    def _process_response(self, content, output_format: str) -> Dict:
        """
        响应处理方法
        """
        try:
            # 调试：将完整返回内容写入 debug.txt 便于查看
            with open("debug.txt", "w", encoding="utf-8") as f:
                f.write(content)
            print("=== Raw Response Content 已保存到 debug.txt ===")

            if output_format == "json":
                content = self._parse_json(content)

                # 获取图表方向（默认横向，如果没有提供）
                orientation = content.get("style", {}).get("orientation", "horizontal")
                if "style" not in content:
                    content["style"] = {}
                content["style"]["orientation"] = orientation
                return content

            return self._safe_execute_code(content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
        except Exception as e:
            return {"error": str(e)}

    def _complete_trend_data(self, data: Dict) -> Dict:
        """补全趋势描述中的缺失值"""
        if "style" not in data:
            data["style"] = {}
        if "estimated_values" not in data["style"]:
            data["style"]["estimated_values"] = []

        estimated_indexes = []

        for series in data["series"]:
            if "trend_info" not in series:
                continue

            trend_infos = series["trend_info"]
            values = series["values"]

            # 确保 trend_info 为列表格式
            if isinstance(trend_infos, dict):
                trend_infos = [trend_infos]

            # 处理每个趋势信息
            for trend in trend_infos:
                # 线性插值补全
                if trend["type"] in ("increase", "decrease"):
                    if "time_range" not in trend or len(trend["time_range"]) < 2:
                        continue

                    try:
                        start_idx = data["categories"].index(trend["time_range"][0])
                        end_idx = data["categories"].index(trend["time_range"][1])

                        # 获取已知的起点和终点值
                        start_val = values[start_idx]
                        end_val = values[end_idx]

                        if start_val is not None and end_val is not None:
                            # 线性插值
                            step = (end_val - start_val) / (end_idx - start_idx)
                            for i in range(start_idx + 1, end_idx):
                                if values[i] is None:
                                    values[i] = start_val + step * (i - start_idx)
                                    estimated_indexes.append(i)
                    except (ValueError, IndexError) as e:
                        print(f"插值错误: {e}, time_range: {trend['time_range']}")
                        continue

        # 标记估算值
        data["style"]["estimated_values"] = estimated_indexes
        return data

    def _parse_json(self, content: str) -> Dict:
        """
        JSON解析方法，自动提取 markdown 格式中的 JSON 部分
        """
        # 尝试提取第一个 ```json ... ``` 包裹的 JSON 内容
        json_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
        match = json_pattern.search(content)
        if match:
            json_str = match.group(1)
        else:
            # 如果没有找到 markdown 包裹，则尝试匹配最外层的 { ... }
            json_pattern = re.compile(r"(\{.*\})", re.DOTALL)
            match = json_pattern.search(content)
            if match:
                json_str = match.group(1)
            else:
                raise ValueError("No valid JSON block found in response.")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            print(f"导致错误的JSON字符串: {json_str[:100]}...")
            raise ValueError(f"Extracted JSON is invalid: {e}")

        # 验证JSON数据格式
        required_keys = {"categories", "series"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            print(f"❌ JSON数据缺少必要字段: {missing}")
            raise ValueError(f"Missing required keys in JSON data: {missing}")

        # 验证series格式
        if not isinstance(data["series"], list):
            print("❌ 'series'字段不是列表类型")
            raise ValueError("'series' field must be a list")

        # 验证categories格式
        if not isinstance(data["categories"], list):
            print("❌ 'categories'字段不是列表类型")
            raise ValueError("'categories' field must be a list")

        # 确保每个系列都有必要的字段
        for i, series in enumerate(data["series"]):
            if not isinstance(series, dict):
                print(f"❌ series[{i}]不是字典类型")
                raise ValueError(f"series[{i}] must be a dictionary")

            if "label" not in series:
                print(f"❌ series[{i}]缺少'label'字段")
                series["label"] = f"Series {i + 1}"

            if "values" not in series:
                print(f"❌ series[{i}]缺少'values'字段")
                series["values"] = [None] * len(data["categories"])

        try:
            # 尝试绘制图表
            self._plot_from_json(data)
        except Exception as e:
            print(f"⚠️ 图表绘制失败，但继续处理: {str(e)}")

        return data

    def parse_chartvlm_csv(self, tsv_text):
        import csv, io

        # 替换转义字符为真实字符
        tsv_clean = tsv_text.replace("\\t", "\t").replace("\\n", "\n").strip()

        # 使用 DictReader 读取 TSV 数据
        reader = csv.DictReader(io.StringIO(tsv_clean), delimiter="\t")
        parsed = {"categories": [], "series": []}  # 更新为列表结构

        # 逐行解析数据
        for row in reader:
            category = None
            for header in row.keys():
                if row[header].strip():  # 找到第一个有数据的列作为类别
                    category = row[header]
                    break

            if category is None:
                raise ValueError("没有找到有效的类别数据")

            # 将类别加入到 "categories"
            parsed["categories"].append(category.strip())

            # 遍历每一列（跳过类别列），将其他列的数据作为 series 的值
            for label, value in row.items():
                if label.strip().lower() in ["year", "category"]:  # 跳过类别列
                    continue

                label_clean = label.strip()
                # 检查是否已存在该系列
                series_found = False
                for series in parsed["series"]:
                    if series["label"] == label_clean:
                        series["values"].append(self.clean_value(value))
                        series_found = True
                        break

                # 如果该系列还没有记录，创建一个新的系列
                if not series_found:
                    parsed["series"].append({
                        "label": label_clean,
                        "values": [self.clean_value(value)]
                    })

        return parsed

    def increment_counter(self):
        """递增计数器并保存到文件中"""
        self.data_counter += 1
        with open(self.counter_file_path, "w") as f:
            f.write(str(self.data_counter))

    def preprocess_and_update(self, ds_json, chartvlm_data):
        """
        替代原来的 update_ds_json_with_chartvlm 方法。
        - 处理 deepseek 中同一 label 出现多次（多个 trend_info）的情况
        - 按照 trend_info 检查是否与 chartvlm 的趋势一致，若一致则替换 values
        """
        from copy import deepcopy

        # 如果 chartvlm_data 为空，直接返回原始数据
        if chartvlm_data is None:
            print("⚠️ ChartVLM 数据为空，返回原始 deepseek 数据")
            return ds_json

        merged = {
            "title": ds_json.get("title", ""),
            "categories": ds_json.get("categories", []),
            "x_label": ds_json.get("x_label", ""),
            "y_label": ds_json.get("y_label", ""),
            "chart_type": ds_json.get("chart_type", "bar"),
            "series": [],
            "style": ds_json.get("style", {
                "orientation": "vertical",
                "color_palette": [],
                "estimated_values": []
            }),
        }

        try:
            # 构建 label → values 映射
            chartvlm_series_dict = {}
            for s in chartvlm_data.get("series", []):
                if "label" in s and "values" in s:
                    chartvlm_series_dict[s["label"].strip().lower()] = s["values"]

            print("✅ chartvlm_series_dict keys:", list(chartvlm_series_dict.keys()))

            # 如果没有有效的数据，返回原始数据
            if not chartvlm_series_dict:
                print("⚠️ ChartVLM 中没有有效的 series 数据，返回原始 deepseek 数据")
                return ds_json

            for ds_series in ds_json.get("series", []):
                print("\n🔍 Processing DS series:", ds_series)
                if "label" not in ds_series:
                    print("⚠️ Series 缺少 label，跳过")
                    merged["series"].append(ds_series)
                    continue

                ds_label = ds_series["label"].strip().lower()
                trend_infos = ds_series.get("trend_info", [])

                # 保证 trend_info 为 list
                if isinstance(trend_infos, dict):
                    trend_infos = [trend_infos]
                elif not isinstance(trend_infos, list):
                    trend_infos = []

                # 查找对应的 chartvlm 值（忽略大小写匹配）
                true_values = None
                for k, v in chartvlm_series_dict.items():
                    if k == ds_label:
                        true_values = [self.clean_value(val) for val in v]
                        break

                # 匹配不到就跳过
                if not true_values:
                    print(f"⚠️ 无法匹配 label: {ds_label}，跳过。")
                    merged["series"].append(ds_series)
                    continue

                # 检查 trend 是否匹配
                matched = False

                # 如果没有趋势信息，直接使用 chartvlm 值
                if not trend_infos:
                    matched = True
                else:
                    for trend in trend_infos:
                        if "time_range" not in trend or "type" not in trend:
                            continue

                        time_range = trend.get("time_range", [])
                        if not time_range or len(time_range) < 2:
                            continue

                        trend_type = trend.get("type", "")

                        try:
                            start_idx = merged["categories"].index(time_range[0])
                            end_idx = merged["categories"].index(time_range[1]) + 1

                            # 确保索引在有效范围内
                            if start_idx < 0 or end_idx > len(true_values) or start_idx >= end_idx:
                                continue

                            print(f"📈 检查趋势：{ds_series['label']} | 索引范围: {start_idx}-{end_idx - 1}")
                            sliced = true_values[start_idx:end_idx]
                            predicted_trend = self.detect_trend(sliced)
                            print(f"🔍 predicted: {predicted_trend}, expected: {trend_type}")
                            if predicted_trend.startswith(trend_type.replace("rapid_", "")):
                                matched = True
                                break
                        except Exception as e:
                            print(f"⚠️ 趋势匹配失败 {ds_series['label']}：{e}, time_range: {time_range}")
                            continue

                if matched:
                    new_series = deepcopy(ds_series)
                    new_series["values"] = true_values
                    merged["series"].append(new_series)
                    print(f"✅ 替换 {ds_series['label']}（趋势匹配）")
                else:
                    merged["series"].append(ds_series)
                    print(f"❌ 保留 {ds_series['label']}（趋势不符或无匹配）")

            return merged

        except Exception as e:
            print(f"❌ preprocess_and_update 错误: {str(e)}")
            return ds_json  # 出错时返回原始数据

    def detect_trend(self, values):
        """简单趋势检测（升/降/波动）"""
        clean_values = [v for v in values if isinstance(v, (int, float))]
        if len(clean_values) < 2:
            return "stable"
        if all(x <= y for x, y in zip(clean_values, clean_values[1:])):
            return "increase"
        elif all(x >= y for x, y in zip(clean_values, clean_values[1:])):
            return "decrease"
        else:
            return "fluctuate"

    def clean_value(self, val):
        """清理和转换值为浮点数，处理特殊格式"""
        if val is None:
            return None

        # 如果是字符串，尝试提取数值
        if isinstance(val, str):
            val = val.strip()

            # 处理估计值标记
            if val.startswith("[est]"):
                val = val[5:].strip()

            # 处理百分比
            if val.endswith("%"):
                val = val[:-1].strip()

            # 提取第一个数字序列
            match = re.search(r'[-+]?\d*\.?\d+', val)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
            return None

        # 如果是数值，直接转换
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _plot_from_json(self, data: Dict):
        """
        从 JSON 生成柱状图，支持：
        - 横向/纵向显示
        - 缺失值高亮显示
        - 估算值特殊标记
        - 自动处理包含字符串的数值数据
        """
        import numpy as np

        # 数据清洗函数 - 处理可能包含 "[est]数值" 字符串的数据
        def clean_series(series):
            for entry in series:
                cleaned_values = []
                for v in entry["values"]:
                    if isinstance(v, str):
                        # 提取数字，比如 "[est] 81" -> 81
                        match = re.search(r'\d+', v)
                        if match:
                            cleaned_values.append(int(match.group()))
                        else:
                            cleaned_values.append(None)
                    else:
                        cleaned_values.append(v)
                entry["values"] = cleaned_values
            return series

        data["series"] = clean_series(data["series"])

        # 初始化图表
        plt.figure(figsize=(12, 6))
        categories = data["categories"]
        num_categories = len(categories)
        y_pos = np.arange(num_categories)
        height = 0.35  # 柱状图宽度
        series = data["series"]
        colors = data.get("style", {}).get("color_palette", ["#1f77b4", "#ff7f0e", "#2ca02c"])
        orientation = data["style"].get("orientation", "horizontal")
        estimated_indexes = data.get("style", {}).get("estimated_values", [])

        # 遍历每个数据系列
        for idx, series_item in enumerate(series):
            values = series_item["values"]
            label = series_item["label"]
            color = colors[idx % len(colors)]
            offset = (idx - (len(series) - 1) / 2) * height  # 多系列偏移量

            # 预处理数据：分离正常值、缺失值和估算值
            clean_values = []
            missing_mask = []
            estimate_mask = []

            for i, val in enumerate(values):
                if val is None:
                    clean_values.append(0)  # 缺失值设为0以便显示
                    missing_mask.append(True)
                    estimate_mask.append(False)
                else:
                    clean_values.append(val)
                    missing_mask.append(False)
                    estimate_mask.append(i in estimated_indexes)

            # 柱状图绘制
            if orientation == "horizontal":
                bars = plt.barh(y_pos + offset, clean_values, height,
                                color=color, label=label, alpha=0.7)

                # 标记估算柱
                for i, is_estimate in enumerate(estimate_mask):
                    if is_estimate:
                        bars[i].set_hatch('xx')
                        bars[i].set_edgecolor(color)
                        bars[i].set_alpha(0.5)
            else:
                bars = plt.bar(y_pos + offset, clean_values, height,
                               color=color, label=label, alpha=0.7)
                for i, is_estimate in enumerate(estimate_mask):
                    if is_estimate:
                        bars[i].set_hatch('xx')
                        bars[i].set_edgecolor(color)
                        bars[i].set_alpha(0.5)

            # 高亮缺失值（红色半透明条）
            for i, is_missing in enumerate(missing_mask):
                if is_missing:
                    if orientation == "horizontal":
                        plt.barh(y_pos[i] + offset, 2, height,
                                 color='red', alpha=0.5, hatch='//')
                    else:
                        plt.bar(y_pos[i] + offset, 2, height,
                                color='red', alpha=0.5, hatch='//')

        # 添加图表装饰
        if orientation == "horizontal":
            plt.xlabel(data.get("y_label", "Value"))
            plt.ylabel(data.get("x_label", "Category"))
            plt.yticks(y_pos, categories)
        else:
            plt.ylabel(data.get("y_label", "Value"))
            plt.xlabel(data.get("x_label", "Category"))
            # 根据标签长度自动调整旋转角度
            rotation = 45 if any(len(cat) > 5 for cat in categories) else 0
            plt.xticks(y_pos, categories, rotation=rotation, ha='right' if rotation else 'center')

        # 添加标题和图例
        plt.title(data.get("title", "Student Answer Visualization"))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 自动调整布局
        plt.tight_layout()

        # 保存和显示图像
        data_path = os.path.join(self.data_save_folder, f"answer{self.data_counter}.png")
        plt.savefig(data_path, dpi=300, bbox_inches='tight')

        # Streamlit 显示
        # buf = BytesIO()
        # plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        # buf.seek(0)
        # st.image(buf, caption=data.get("title", "Student Graph"), use_container_width=True)
        print(f"Data saved to: {os.path.abspath(data_path)}")

    def _safe_execute_code(self, content: str) -> Dict:
        """
        安全执行代码方法
        """
        code_block = self._extract_code(content)
        if not code_block:
            return {"error": "No valid code block"}
        try:
            restricted_globals = self.safe_modules.copy()
            local_env = {}
            exec(code_block, restricted_globals, local_env)
            plt.show()
            return {"status": "success", "output": local_env.get("figure")}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}

    @staticmethod
    def _extract_code(content: str) -> Optional[str]:
        """
        提取代码块方法
        """
        match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else None


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    try:
        # 输入参数配置
        initial_instruction = (
            "Now I'll send you the Requirement, graph and Sample answer of the first Writing question of IELTS Academic. "
            "You need to learn how to reverse generate the graph according to the requirement and given answer which "
            "describes the graph by the materials I give to you."
        )

        requirement = (
            "The chart below shows the amount of leisure time enjoyed by men and women of different employment status."
            "Write a report for a university lecturer describing the information shown below."
            "Leisure time in a typical week in hour: by sex and employment status, 1998-99."
        )

        student_answer = ('''The bar chart illustrates the average number of leisure hours per week enjoyed by men and women in different employment categories in the years 1998–1999. The categories include full-time employment, part-time employment, unemployment, retirement, and housewives.
Overall, men tended to have more leisure time than women in most employment statuses. However, data for part-time workers and housewives is only available for females.
Among the unemployed and retired, both men and women enjoyed the most leisure time, with unemployed men averaging around 85 hours per week, slightly more than unemployed women at approximately 78 hours. A similar pattern is seen among the retired group, with men enjoying about 83 hours and women around 78 hours.
Full-time employed men had around 45 hours of leisure time, compared to about 38 hours for women in the same category. Women working part-time had approximately 40 hours of free time, while housewives had even more, averaging 50 hours per week.
In conclusion, those who were not working (either unemployed or retired) had the most leisure time, with men consistently having slightly more free time than women across similar categories.
''')

        # 初始化生成器
        generator = GraphGenerator()

        # 检查TSV数据有效性
        tsv_text = """Characteristic \t Male \t Female \n Employed (Full Time) \t 44 \t 38 \n Employed (Part Time) \t 85 \t 40 \n Unemployed \t 78 \t 78 \n Retired \t 83 \t 78 \n Housewives \t 50 \t 50 \n"""
        try:
            chartvlm_data = generator.parse_chartvlm_csv(tsv_text)
            print("✅ ChartVLM数据解析成功")
        except Exception as e:
            print(f"❌ ChartVLM数据解析失败: {str(e)}")
            chartvlm_data = None

        # 执行API调用
        image_path = "data/bar2.png"
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"⚠️ 警告: 图片不存在 '{image_path}'")

        result = generator.call_gpt_and_generate(
            initial_instruction=initial_instruction,
            requirement=requirement,
            student_answer=student_answer,
            image_path=image_path,
            output_format="json",
            chartvlm_data=chartvlm_data
        )

        # 处理结果
        if "error" in result:
            print(f"❌ 错误: {result['error']}")
        else:
            print("✅ 生成数据成功:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ 程序执行错误: {str(e)}")
        import traceback

        traceback.print_exc()
