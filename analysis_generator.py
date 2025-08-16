from config import model, MAX_RESPONSE_LENGTH

def generate_comprehensive_analysis(paper_content, question=None, max_length=MAX_RESPONSE_LENGTH):
    """Generate comprehensive analysis of the research paper with improved structure"""
    
    if question:
        prompt = f"""
        Based on the research paper content below, provide a detailed answer to the question: "{question}"
        
        Requirements:
        - Provide a comprehensive response (maximum {max_length} words)
        - Structure your response with clear headings and subheadings
        - Use bullet points for key findings or multiple items
        - Include specific details and evidence from the paper
        - Be precise and technical where appropriate
        - Format the response for easy reading with proper paragraphs
        
        Paper content: {paper_content[:5000]}
        
        Question: {question}
        """
    else:
        prompt = f"""
        Analyze this research paper and provide a comprehensive summary with the following structure:
        
        RESEARCH OBJECTIVE:
        What problem does this paper address? What are the main research questions?
        
        METHODOLOGY:
        How did the researchers approach the problem? What methods and techniques were used?
        
        KEY FINDINGS:
        What are the main results and discoveries? Present these as clear bullet points.
        
        SIGNIFICANCE AND CONTRIBUTIONS:
        Why is this research important? What are the key contributions to the field?
        
        LIMITATIONS AND CHALLENGES:
        What are the acknowledged limitations or challenges faced?
        
        FUTURE WORK:
        What directions for future research are suggested?
        
        CONCLUSION:
        Summarize the overall impact and importance of this work.
        
        Keep the response detailed but within {max_length} words. Use proper paragraph structure and bullet points where appropriate.
        
        Paper content: {paper_content[:5000]}
        """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Analysis generation failed: {str(e)}"