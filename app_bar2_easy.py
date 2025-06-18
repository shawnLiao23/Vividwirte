import os
import streamlit as st
from PIL import Image
from bar import GraphGenerator
from scoring import WritingScorer

def show():
    # 初始化
    generator = GraphGenerator()

    # 输入instruction
    initial_instruction = (
        "Now I'll send you the Requirement, graph and Sample answer of the first Writing question of IELTS Academic. "
        "You need to learn how to reverse generate the graph according to the requirement and given answer which "
        "describes the graph by the materials I give to you."
    )

    # 页面标题
    st.title("🎓 IELTS Task 1 Graph Generation System")

    # 页面布局样式设置
    st.markdown("""
        <style>
        .block-container {
            max-width: 1500px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # 创建左右两栏
    left_col, right_col = st.columns([1, 1])

    with right_col:
        st.header("📈 Student Graph")
        graph_placeholder = st.empty()
        graph_placeholder.markdown("Your graph would be displayed after the generation.")

        st.header("💡 Writing Suggestions")
        evaluation_placeholder = st.empty()
        evaluation_placeholder.markdown("Your writing evaluation will appear here after submission.")
        
    with left_col:
        st.header("📋 Requirement")
        requirement = (
            "The chart below shows the amount of leisure time enjoyed by men and women of different employment status."
"Write a report for a university lecturer describing the information shown below."
"Leisure time in a typical week in hour: by sex and employment status, 1998-99."
        )
        st.markdown(requirement)

        # 显示原图
        st.subheader("📊 Original Image")
        img = Image.open("data/bar2.png")
        st.image(img, caption="Original Graph", use_container_width=True)

        st.subheader("👉 Recommend Structure")
        structure = ('''
                    The bar chart illustrates the average number of leisure hours per week enjoyed by men and women in different employment categories in the years 1998–1999. The categories include full-time employment, part-time employment, unemployment, retirement, and housewives.
Overall, men tended to have more leisure time than women in most employment statuses. However, data for part-time workers and housewives is only available for females.
Among the unemployed and retired, both men and women enjoyed the most leisure time, with unemployed men averaging around ____ hours per week, slightly more than unemployed women at approximately ___ hours. A similar pattern is seen among the retired group, with men enjoying about ___ hours and women around 78 hours.
_____ employed men had around ___ hours of leisure time, compared to about 38 hours for women in the same category. Women working part-time had approximately 40 hours of free time, while housewives had even more, averaging ___ hours per week.
In conclusion, those who were not working (either unemployed or retired) had the most leisure time, with men consistently having slightly more free time than women across similar categories.
                     ''')
        st.markdown(structure)

        # 学生作文输入框
        st.subheader("✍️ Student Answer")
        student_answer = st.text_area("Please write your answer here:", height=300)
        text_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.txt")

        # 保存学生答案为.txt文件
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(student_answer)

        tsv_text = """Characteristic \t Male \t Female \n Employed (Full Time) \t 44 \t 38 \n Employed (Part Time) \t 85 \t 40 \n Unemployed \t 78 \t 78 \n Retired \t 83 \t 78 \n Housewives \t 50 \t 50 \n"""
        chartvlm_data = generator.parse_chartvlm_csv(tsv_text)

        # 按钮触发生成图像
        if st.button("🚀 Generate Graph from Student Answer"):
            if not student_answer.strip():
                st.warning("Please enter the student's answer before generating.")
            else:
                with st.spinner("Generating graph and evaluating writing..."):
                    result = generator.call_gpt_and_generate(
                        initial_instruction=initial_instruction,
                        requirement=requirement,
                        student_answer=student_answer,
                        image_path="data/bar2.png",
                        output_format="json",
                        chartvlm_data=chartvlm_data
                    )

                    graph_placeholder.empty()
                    img_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.png")
                    graph_placeholder.image(img_path, caption="Student Graph", use_container_width=True)
                    # 评分
                    evaluation_placeholder.empty()
                    scorer = WritingScorer()
                    evaluation = scorer.evaluate_writing(student_answer)
                    evaluation_placeholder.text(evaluation)
                    generator.increment_counter()

                # 图像生成结果提示
                if "error" in result:
                    st.error(f"Generation failed: {result['error']}")
                else:
                    st.success("✅ Student-based chart generated successfully!")

