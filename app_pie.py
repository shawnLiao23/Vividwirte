import os
import streamlit as st
from PIL import Image
from pie import GraphGenerator
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

    # 页面布局
    st.markdown("""
        <style>
        .block-container {
            max-width: 1500px;  /* 设置最大宽度 */
            padding-left: 2rem;  /* 设置左侧内边距 */
            padding-right: 2rem;  /* 设置右侧内边距 */  
        }
    """, unsafe_allow_html=True)

    # 创建左右两栏
    left_col, right_col = st.columns([1,1])

    with right_col:
        st.header("📈 Student Graph")
        graph_placeholder = st.empty()
        graph_placeholder.markdown("Your graph would be displayed after the generation.")

        st.header("💡 Writing Suggestions")
        evaluation_placeholder = st.empty()
        evaluation_placeholder.markdown("Your writing evaluation will appear here after submission.")

    with left_col:
        st.header("📋 Requirement") # 展示题目要求
        requirement = (
            """The pie chart below shows the proportion of different categories of families living in poverty in UK in 2002.
            Summarise the information by selecting and reporting the main features, and make comparisons where relevant.
            Write at least 150 words."""
            )
        st.markdown(requirement)

        # 显示原图
        st.subheader("📊 Original Image")
        img = Image.open("data/pie.png")
        st.image(img, caption="Original Graph", use_container_width=True)

        # 学生作文输入框
        st.subheader("✍️ Student Answer")
        student_answer = st.text_area("Please write your answer here:", height=300)
        text_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.txt")
        # 保存学生答案为.txt文件
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(student_answer)

        deplot_txt = "TITLE | Proportion of people from each household type living in poverty<0x0A>All households<0x0A>14% | Single aged person<0x0A>7% <0x0A> Aged couple<0x0A>5% | 5% <0x0A> Couple with children<0x0A>15% | 15% <0x0A> Single, no children<0x0A>24% | 24% <0x0A> Sole parent<0x0A>26% | 26% <0x0A> Couple, no children<0x0A>9% | 9%"

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
                        image_path="data/pie.png",
                        output_format="json",
                        deplot_txt=deplot_txt
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
