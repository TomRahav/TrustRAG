from openai import OpenAI



class GPT():
    def __init__(self):
        api_keys = "your-api-keys"
        self.max_output_tokens = 1024
        self.client = OpenAI(api_key=api_keys)

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response