import openai
import os
openai.api_key = os.environ["OPENAI_API_KEY"]
import time

query = "Feel like eating some good chinese food tonight."

prompt = f"""
VITALLY IMPORTANT: Ensure that your response is pure python, without any other non-python text.
I have the following tables (as a csv format). 

User_profile (file_tables/mock_data/user_profile.csv): 
user_id,user_name,user_preferences,max_budget,location_long,location_lat 
"U123456","John Doe","Italian, Mexican",30,-119.13142,46.23511 

Restaurants (file_tables/mock_data/user_profile.csv): 
store,location_long,location_lat,genre,price 
"Restaurant A",-119.1266,46.1923,"Italian",35 

Menus (file_tables/mock_data/menu.csv): 
store,genre,price,food_name,food_desc 
"Restaurant A","Italian",35,"Margherita Pizza","Traditional Neapolitan-style pizza topped with tomato sauce, mozzarella cheese, and fresh basil leaves." 

Write a python script that uses pandas 
(and other libraries as needed) to answer the question: 
{query}.

Avoid merging or joining tables when possible!!!

Your response should follow the class definition as follows:

class agent_query():
    def __init__(self):
        self.name = "yes"

    @staticmethod
    def run_query():
        ### < INSERT QUERY HERE >
        return <ANSWER HERE>

VITALLY IMPORTANT: Ensure that your response is pure python, without any other non-python text.
I REPEAT: Ensure that your response is pure python, DO NOT INCLUDE non-python text.
"""

def run_program():

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    response = response.choices[0].message.content

    def save_agent_response_to_file(string):
        with open("agent_code.py", 'w') as file:
            file.write(string)

    save_agent_response_to_file(response)

    from agent_code import agent_query

    try:
        result = agent_query.run_query()
        return result
    except Exception as e:
        print(f"Error encountered: {e}")
        print("Retrying...")
        time.sleep(2)  # Add a small delay before rerunning
        return run_program()

output = run_program()
print(output)
