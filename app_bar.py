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
            "The bar chart below shows the total number of minutes (in billions) of telephone calls in Australia,"
            "divided into three categories, from 2001- 2008. Summarise the information by selecting and reporting the"
            "main features and make comparisons where relevant. Write at least 150 words."
        )
        st.markdown(requirement)

        # 显示原图
        st.subheader("📊 Original Image")
        img = Image.open("data/bar.png")
        st.image(img, caption="Original Graph", use_container_width=True)

        # 学生作文输入框
        st.subheader("✍️ Student Answer")
        student_answer = st.text_area("Please write your answer here:", height=300)
        text_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.txt")

        # 保存学生答案为.txt文件
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(student_answer)

        tsv_text = """Year \t Local fixed line calls \t National and international fixed line calls \t Mobile calls \n 2001 \t 72 \t 38 \t 3 \n 2002 \t 78 \t 40 \t 6 \n 2003 \t 83 \t 42 \t 10 \n 2004 \t 88 \t 45 \t 12 \n 2005 \t 90 \t 47 \t 15 \n 2006 \t 85 \t 50 \t 23 \n 2007 \t 78 \t 52 \t 38 \n 2008 \t 73 \t 58 \t 48 \n"""
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
                        image_path="data/bar.png",
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
