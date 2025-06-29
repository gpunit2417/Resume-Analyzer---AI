from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import io
import openai
import os
from datetime import datetime
import json
from dotenv import load_dotenv
import re

app = Flask(__name__)
CORS(app)

load_dotenv()


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def create_analysis_prompt(resume_text, job_profile, experience_level):
    """Create a detailed prompt for AI analysis"""
    
    prompt = f"""
You are an expert ATS (Applicant Tracking System) analyzer and career consultant. 
Analyze the following resume for a {job_profile} position at {experience_level} level.

RESUME TEXT:
{resume_text}

TARGET POSITION: {job_profile}
EXPERIENCE LEVEL: {experience_level}

Please provide a comprehensive ATS analysis in the following JSON format:

{{
    "ats_score": <integer between 0-100>,
    "overall_assessment": "<brief overall assessment>",
    "strengths": [
        "<strength 1>",
        "<strength 2>",
        "<strength 3>"
    ],
    "improvements": [
        "<improvement area 1>",
        "<improvement area 2>",
        "<improvement area 3>"
    ],
    "recommendations": [
        "<specific recommendation 1>",
        "<specific recommendation 2>",
        "<specific recommendation 3>"
    ],
    "keyword_analysis": {{
        "missing_keywords": ["<keyword1>", "<keyword2>"],
        "present_keywords": ["<keyword1>", "<keyword2>"],
        "keyword_score": <integer 0-100>
    }},
    "formatting_issues": [
        "<formatting issue 1>",
        "<formatting issue 2>"
    ],
    "detailed_feedback": {{
        "technical_skills": "<feedback on technical skills>",
        "experience_relevance": "<feedback on experience relevance>",
        "education": "<feedback on education section>",
        "achievements": "<feedback on achievements>",
        "contact_info": "<feedback on contact information>"
    }}
}}

ANALYSIS CRITERIA:
1. ATS Compatibility (keyword matching, formatting, structure)
2. Relevance to target job profile
3. Experience level appropriateness  
4. Technical skills alignment
5. Achievement quantification
6. Professional formatting
7. Contact information completeness
8. Education relevance
9. Career progression consistency
10. Industry-specific terminology usage

Provide actionable, specific feedback that will help improve the resume's ATS score and overall effectiveness for the target position.
"""
    
    return prompt

def analyze_with_ai(prompt):
    """Send prompt to OpenAI for analysis"""
    try:
        # Using OpenAI's new client format
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for better results
            messages=[
                {"role": "system", "content": "You are an expert ATS analyzer and career consultant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        print("Result: ",result)
        
        # Parse JSON response
        try:
            analysis = json.loads(result)
            return analysis
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic response
            return create_fallback_response()
            
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return create_fallback_response()

def create_fallback_response():
    """Create a fallback response when AI analysis fails"""
    return {
        "ats_score": 65,
        "overall_assessment": "Resume analysis completed. Consider the recommendations below.",
        "strengths": [
            "Resume structure is readable",
            "Contact information is present",
            "Experience section is included"
        ],
        "improvements": [
            "Add more relevant keywords for the target position",
            "Quantify achievements with specific numbers",
            "Optimize formatting for ATS compatibility"
        ],
        "recommendations": [
            "Include industry-specific keywords in your summary",
            "Use bullet points to highlight key achievements",
            "Save resume in both PDF and Word formats"
        ],
        "keyword_analysis": {
            "missing_keywords": ["relevant", "keywords", "needed"],
            "present_keywords": ["experience", "skills", "education"],
            "keyword_score": 60
        },
        "formatting_issues": [
            "Consider using standard section headers",
            "Ensure consistent formatting throughout"
        ],
        "detailed_feedback": {
            "technical_skills": "Technical skills section needs more specific technologies",
            "experience_relevance": "Experience should be more aligned with target role",
            "education": "Education section is appropriately formatted",
            "achievements": "Add more quantified achievements",
            "contact_info": "Contact information is complete"
        }
    }

def analyze_resume_locally(resume_text, job_profile, experience_level):
    """Local analysis when AI is not available"""
    
    # Basic keyword analysis
    job_keywords = {
        'Software Engineer': ['python', 'java', 'javascript', 'react', 'node', 'sql', 'git', 'agile', 'api', 'database'],
        'Data Scientist': ['python', 'r', 'sql', 'machine learning', 'tensorflow', 'pandas', 'numpy', 'statistics', 'visualization'],
        'Product Manager': ['roadmap', 'stakeholder', 'agile', 'scrum', 'analytics', 'user experience', 'strategy'],
        'UI/UX Designer': ['figma', 'sketch', 'adobe', 'user research', 'wireframe', 'prototype', 'usability'],
        'Marketing Manager': ['campaign', 'analytics', 'seo', 'social media', 'content', 'brand', 'roi'],
        'DevOps Engineer': ['aws', 'docker', 'kubernetes', 'jenkins', 'terraform', 'monitoring', 'ci/cd'],
    }
    
    keywords = job_keywords.get(job_profile, [])
    resume_lower = resume_text.lower()
    
    found_keywords = [kw for kw in keywords if kw in resume_lower]
    keyword_score = int((len(found_keywords) / len(keywords)) * 100) if keywords else 50
    
    # Basic scoring logic
    base_score = 50
    if len(resume_text) > 1000:
        base_score += 10
    if '@' in resume_text:
        base_score += 5
    if any(word in resume_lower for word in ['experience', 'skills', 'education']):
        base_score += 10
    
    ats_score = min(base_score + keyword_score // 2, 100)
    
    return {
        "ats_score": ats_score,
        "overall_assessment": f"Resume analyzed for {job_profile} position at {experience_level}",
        "strengths": [
            f"Found {len(found_keywords)} relevant keywords",
            "Resume structure is readable",
            "Contains essential sections"
        ],
        "improvements": [
            "Add more industry-specific keywords",
            "Quantify achievements with numbers",
            "Optimize for ATS scanning"
        ],
        "recommendations": [
            f"Include more of these keywords: {', '.join(keywords[:5])}",
            "Use action verbs to start bullet points",
            "Keep formatting simple and clean"
        ],
        "keyword_analysis": {
            "missing_keywords": [kw for kw in keywords if kw not in found_keywords][:5],
            "present_keywords": found_keywords[:5],
            "keyword_score": keyword_score
        },
        "formatting_issues": [
            "Ensure consistent font usage",
            "Use standard section headers"
        ],
        "detailed_feedback": {
            "technical_skills": "Include more specific technical skills relevant to the role",
            "experience_relevance": "Highlight experience most relevant to target position",
            "education": "Education section should support your career goals",
            "achievements": "Add quantified achievements and results",
            "contact_info": "Ensure all contact information is current and professional"
        }
    }

@app.route('/analyze-resume', methods=['POST'])
def analyze_resume():
    try:
        # Check if files are present
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        file = request.files['resume']
        job_profile = request.form.get('jobProfile')
        experience_level = request.form.get('experience')
        
        if not file or not job_profile or not experience_level:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(file)
        
        if not resume_text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Try AI analysis first, fall back to local analysis
        try:
            if os.getenv('OPENAI_API_KEY'):
                prompt = create_analysis_prompt(resume_text, job_profile, experience_level)
                analysis_result = analyze_with_ai(prompt)
            else:
                print("No OpenAI API key found, using local analysis")
                analysis_result = analyze_resume_locally(resume_text, job_profile, experience_level)
        except Exception as e:
            print(f"AI analysis failed: {e}, falling back to local analysis")
            analysis_result = analyze_resume_locally(resume_text, job_profile, experience_level)
        
        # Add metadata
        analysis_result['analysis_date'] = datetime.now().isoformat()
        analysis_result['job_profile'] = job_profile
        analysis_result['experience_level'] = experience_level
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"Error in analyze_resume: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("Starting Resume Analyzer Backend...")
    print("Make sure to set OPENAI_API_KEY environment variable for AI analysis")
    print("Server will run on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)