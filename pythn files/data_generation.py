# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
from together import Together
import os
import pandas as pd
import time
import random
from datetime import datetime

# Set your Together.ai API key
os.environ["TOGETHER_API_KEY"] = "6b524c1f4c5137249157731767ae0d8483dc6cfb3c503655aa97f6ea3eeaff30"
client = Together()

# Lists provided by the user
names = ["Aaditya", "Aayu", "Bhuv", "Brij", "Chahal", "Chirag", "Daiwik", "Dipal", "Eashan", "Ehan", "Fanish", "Grishm", "Gyan", "Hans", "Hiral", "Ichaa", "Ilesh", "Jagrav", "Jivin", "Kairav", "Kanha", "Kripal", "Kunal", "Lahar", "Mahin", "Malay", "Manav", "Naksh", "Nakul", "Niraj", "Ojas", "Ovin", "Palash", "Praneel", "Raahi", "Rijul", "Ritul", "Saanjh", "Shray", "Snehal", "Taksha", "Tuhin", "Udanth", "Uttam", "Ujjwal", "Vaishnav", "Viral", "Wriddhish", "Yagna", "Yash", "Zev", "Aanya", "Amisha", "Bani", "Brinda", "Chaaya", "Charvi", "Dhanvi", "Disha", "Ekta", "Eshani", "Forum", "Gargi", "Gina", "Harini", "Harshini", "Hyma", "Iksha", "Ishani", "Janki", "Kamna", "Kashvi", "Laksha", "Larisa", "Manasi", "Mythily", "Naina", "Namrata", "Nirosha", "Oni", "Oshee", "Pratiti", "Parinita", "Pihu", "Rachna", "Roopali", "Ranjita", "Saanvi", "Sweta", "Tanima", "Trayi", "Trisha", "Upasana", "Upama", "Vanshika", "Varsha", "Veda", "Watika", "Yahvi", "Yashika", "Zoya"]
last_names = ["Raj", "Sharma", "Iyer", "Patel", "Verma", "Desai", "Gupta", "Mehta", "Chowdhury", "Yadav", "Reddy", "Naidu", "Kumar", "Nair", "Shukla", "Singh", "Dixit", "Soni", "Shah", "Joshi", "Kohli", "Ghosh", "Choudhury", "Basu", "Tiwari", "Pandey", "Rai", "Bhatt", "Gupta", "Jain", "Sahu", "Gupta", "Mishra", "Das", "Soni", "Rathore", "Saxena", "Bansal", "Khandelwal", "Garg", "Kapoor", "Bhat", "Kaur", "Chawla", "Chopra", "Agarwal", "Mittal", "Bedi", "Ruparel", "Gulati", "Seth", "Kumari", "Thakur", "Bhardwaj", "Singh", "Raghav"]
languages_known = ["English", "Tamil", "Hindi", "Sanskrit", "Telugu", "Marathi"]
select_reasons = ["Excellent communication skills", "Strong technical knowledge", "Proven leadership qualities", "Adaptability", "High enthusiasm for the role"]
reject_reasons = ["Lack of relevant experience", "Not enough technical expertise", "Inability to communicate clearly", "Unsuitable for the job role", "Low problem-solving abilities"]
designations = {
    "Software Developer": {"domains": ["Web Development", "Mobile Apps"], "skills": ["JavaScript", "React", "Node.js"]},
    "Data Scientist": {"domains": ["Data Analysis", "Machine Learning"], "skills": ["Python", "R", "Machine Learning"]},
    "Project Manager": {"domains": ["Project Management", "Team Leadership"], "skills": ["Agile", "Scrum", "Jira"]}
}
experience_levels = ["Beginner", "Intermediate", "Advanced"]
work_environments = ["Remote", "On-site", "Hybrid"]

# Function to generate a text response based on the given prompt
def generate_text(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": f":{prompt}"}],
    )
    return response.choices[0].message.content

# Generating data for the table
data = []
for i in range(250):  # Adjust the range to generate more candidates if needed
    # Generate ID with first 4 letters of name and first 2 of last name + number
    name = random.choice(names)
    owner = "KotaSai"
    last_name = random.choice(last_names)
    full_name = f"{name} {last_name}"
    id_formatted = f"{owner}{i+1}"

    # Select a random role and job description
    designation = random.choice(list(designations.keys()))
    expected_experience = random.choice(["0-2 years", "3-5 years", "6-8 years", "9+ years"])
    domains_needed = ", ".join(designations[designation]["domains"])
    job_description = f"Expected_experience : {expected_experience}, Domains: {domains_needed}"

    select_status = random.choice(["Select", "Reject"])

    # Randomize experience level and work environment
    experience_level = random.choice(experience_levels)
    work_environment = random.choice(work_environments)

    # Randomly select languages
    selected_languages = random.sample(languages_known, k=random.randint(1, len(languages_known)))

    # Select reasons for decision
    reasons = random.sample(
        select_reasons if select_status == "Select" else reject_reasons,
        2
    )

    # Define prompt templates for transcripts, profiles
    prompt_templates = {
        "Select": {
            "transcript": (
                f"Generate a positive, detailed interview transcript for candidate {full_name} interviewing for the role of {designation}. "
                f"Job description: '{job_description}'. Highlight strengths such as demonstrated skills in {', '.join(designations[designation]['skills'])}, "
                f"excellent problem-solving abilities, enthusiasm, and potential for growth. Emphasize their impressive experience in {domains_needed}. "
                f"Do not generate generic statements like 'Here is an interview transcript for ...'."
            ),
            "profile": (
                f"Generate a detailed dummy resume for candidate {full_name}, who is interviewing for the role of {designation}. "
                f"Include the following sections: Contact Information (Email, Phone), Objective (A brief about the candidate), Skills (List key skills like {', '.join(designations[designation]['skills'])}), "
                f"Experience (Mention relevant roles or projects in {', '.join(domains_needed)}), Education (Degrees or certifications), Languages Known (List of languages known). "
                f"Also include the work environment preference and experience level as: Experience level: {experience_level}, Work environment: {work_environment}. "
                f"Ensure the resume appears realistic and matches the candidate's profile."
            )
        },
        "Reject": {
            "transcript": (
                f"Generate a detailed, constructive interview transcript for candidate {full_name} interviewing for the role of {designation}. "
                f"Job description: '{job_description}'. Mention areas of improvement, such as lacking key technical skills in {', '.join(designations[designation]['skills'])}, "
                f"insufficient experience in the required domains, and limited understanding of job requirements. Highlight reasons like 'Struggled to communicate ideas' or 'Needed improvement in problem-solving skills'. "
                f"Do not generate generic statements like 'Here is an interview transcript for ...'."
            ),
            "profile": (
                f"Generate a detailed dummy resume for candidate {full_name} who is interviewing for the role of {designation}. "
                f"Include sections like: Contact Information (Email, Phone), Objective (State the candidate's desire to work in a related field), Skills (Mention skills like {', '.join(designations[designation]['skills'])} and relevant ones), "
                f"Experience (List past roles or projects and mention skills to be developed), Education (Include any degrees or certifications). "
                f"Also include Work Environment preference (Remote, Hybrid, On-site) and Experience level (beginner, intermediate, advanced). "
                f"Ensure the resume includes areas to improve and is constructive in tone."
            )
        }
    }

    # Generate the transcript and profile based on selection/rejection reason
    reason = reasons[0]
    if select_status == "Select":
        transcript = generate_text(prompt_templates["Select"]["transcript"])
        profile = generate_text(prompt_templates["Select"]["profile"])
    else:
        transcript = generate_text(prompt_templates["Reject"]["transcript"])
        profile = generate_text(prompt_templates["Reject"]["profile"])

    # Prepare the candidate data row
    data.append({
        'ID': id_formatted,
        'Name': full_name,
        'Role': designation,
        'Transcript': transcript,
        'Resume': profile,
        'Performance': select_status,
        'Reason for decision': ", ".join(reasons),
        'Job Description': job_description
    })

# Delay before the next iteration
    time.sleep(5)
# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel("Kota_Venkata_Krishna_Gopi_Krishna_Sai_data.xlsx", index=False)

# %%



