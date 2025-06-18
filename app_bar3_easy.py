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
            """You should spend about 20 minutes on this task.
The chart below gives information about global population percentages and distribution of wealth by region.

Summarise the information by selecting and reporting the main features, and make comparisons where relevant.
Write at least 150 words."""
        )
        st.markdown(requirement)

        # 显示原图
        st.subheader("📊 Original Image")
        img = Image.open("data/bar3.png")
        st.image(img, caption="Original Graph", use_container_width=True)

        st.subheader("👉 Recommend Structure")
        structure = ('''
        The chart provides a detailed comparison of global population and wealth distribution across eight regions, measured in percentages.
        
        Population Distribution
        •	Other Asia Pacific (including countries like Indonesia and Pakistan) represents the largest share at __%, followed closely by India at __%.
        •	China accounts for approximately 18-20%, while ______ contributes 15% to the global population.
        •	Europe and North America each make up around 10%, with Europe slightly higher at 11% and North America at 9%.
        •	Latin America & Caribbean and Rich Asia Pacific (including Japan and South Korea) have the smallest shares, at 5% and 3% respectively.
        Wealth Distribution
        •	___________ holds a staggering 35% of global wealth, the highest among all regions.
        •	_____ follows with 25%, while Rich Asia Pacific accounts for 15-18%.
        •	China and Other Asia Pacific each possess roughly ____% of global wealth.
        •	India, despite its large population, holds only 3% of global wealth, similar to Latin America & Caribbean (4%) and Africa (2%).
        The data reveals a stark imbalance: North America and Europe, with just 20% of the global population, control 60% of the wealth. Meanwhile, India, Africa, and Latin America, representing 40% of the population, possess less than 10% of global wealth. This disparity underscores significant economic inequality across regions.
        ''')
        st.markdown(structure)

        # 学生作文输入框
        st.subheader("✍️ Student Answer")
        student_answer = st.text_area("Please write your answer here:", height=300)
        text_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.txt")

        # 保存学生答案为.txt文件
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(student_answer)

        tsv_text = """Characteristic \t Population \t Global Wealth \n North America \t 5% \t 35% \n China \t 25% \t 2% \n Other Asia Pacific \t 18% \t 2% \n India \t 18% \t 1% \n Europe \t 17% \t 30% \n Africa \t 10% \t 1% \n Latin America & Caribbean \t 8% \t 3% \n Rich Asia Pacific \t 5% \t 25% \n"""
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
                        image_path="data/bar3.png",
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
