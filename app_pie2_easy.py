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
            """The pie chart gives information on UAE government spending in 2000. The total budget was AED 315 billion.
            Summarize the information by selecting and reporting the main features, and make comparisons where relevant.
            Write at least 150 words."""
            )
        st.markdown(requirement)

        # 显示原图
        st.subheader("📊 Original Image")
        img = Image.open("data/pie2.png")
        st.image(img, caption="Original Graph", use_container_width=True)

        st.subheader("👉 Recommend Structure")
        structure = ("""
                The pie chart illustrates the distribution of the UAE government's expenditure in 2000, with a total budget of AED 1______ billion. The spending was allocated across 2______ different sectors, with 3, health and personal social services, and 4 receiving the largest shares.
                The most significant portion of the budget, 5%, was allocated to social security, highlighting the government's focus on welfare and support systems. Health and personal social services accounted for the second-largest share at 6%, reflecting substantial investment in healthcare. 7______ followed closely, receiving 8______% of the budget, which underscores its importance in national development.
                Smaller allocations were made to other sectors. Defence and 9______ each accounted for 7.3% of the budget, while law and order received 10______%. Housing, heritage, and environment were allocated 4.8%, and industry, agriculture, and employment received 4.2%. The smallest share, 2.9%, went to transport, indicating relatively lower priority in this area compared to other sectors.
                In summary, the UAE government's spending in 2000 was heavily directed towards social security, health, and education, with smaller proportions dedicated to defence, infrastructure, and other services. This allocation reflects the government's priorities in welfare and public services during that period.
                """)
        st.markdown(structure)

        # 学生作文输入框
        st.subheader("✍️ Student Answer")
        student_answer = st.text_area("Please write your answer here:", height=300)
        text_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.txt")
        # 保存学生答案为.txt文件
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(student_answer)

        deplot_txt = "TITLE | UAE Government Spending in 2000<0x0A>Health and personal social services<0x0A>16.9% | 16.9% <0x0A> Transport<0x0A>2.9% | 2.9% <0x0A> Social security<0x0A>31.9% | 31.9% <0x0A> Law and order<0x0A>3.4% | 5.4% <0x0A> Education<0x0A>12.1% | 12.1% <0x0A> Industry, agriculture and employment<0x0A>4.2% | 4.2% <0x0A> Defence<0x0A>7.0% | 7.0% <0x0A> Other expenditure<0x0A>7.3% | 7.3% <0x0A> Debt interest<0x0A>7.3% | 7.3%"

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
                        image_path="data/pie2.png",
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
