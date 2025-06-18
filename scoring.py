import os
from openai import OpenAI
from dotenv import load_dotenv

# API配置
# api_key = os.environ.get("OPENAI_API_KEY")
# load_dotenv()  # 加载.env文件
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # 替换为你的实际API密钥
# DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")  # 确认最新API地址

class WritingScorer:
    def __init__(self):
        self.client = OpenAI()
        
    def evaluate_writing(self, student_writing: str):
        system_content = """
        You are an experienced and fair English writing examiner. Your job is to objectively assess English compositions written by students based on consistent and professional standards.

        You must evaluate the writing based on two main categories:
        1. **Grammar**: Consider sentence structure, verb tense consistency, subject-verb agreement, punctuation, and overall grammatical accuracy.
        2. **Vocabulary**: Consider word choice, vocabulary range, precision of expressions, and appropriateness in context.

        For each category, provide:
        - A score from 0 to 10
        - A short explanation of the key issues found in the student's writing
        - Specific suggestions or examples to help the student improve

        Use the following clear format for your feedback:

        [Grammar Score]: X/10  
        [Grammar Issues]: ...  
        [Grammar Suggestions]: ...  

        [Vocabulary Score]: Y/10  
        [Vocabulary Issues]: ...  
        [Vocabulary Suggestions]: ...

        Each score must follow this scale:
        - 10: Excellent with no or only very minor issues
        - 7–9: Good, with some issues that do not significantly affect understanding
        - 4–6: Fair, with noticeable issues or limited variety
        - 1–3: Poor, with frequent errors or basic word use
        - 0: Incomprehensible or completely incorrect

        Remain objective and consistent in your evaluations. Do not be influenced by the creativity or the topic of the essay. Only focus on grammar and vocabulary quality.
        """

        initial_instruction = "You are given an English composition by a student. Assess their performance."
        requirement = "Please give two separate scores: one for Grammar, and one for Vocabulary. Also, provide a brief rationale for each score."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Task Context: {initial_instruction}"},
            {"role": "user", "content": f"Official Requirement: {requirement}"},
            {"role": "user", "content": f"""
    [BEGIN JUDGING]
    {student_writing}
    [END JUDGING]
            """}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=1500
        )
        result = response.choices[0].message.content.strip()
        return result
