import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
import traceback
import warnings
import io
import sys
import re

llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,  # Creativity level
        api_key=""  # Replace with your actual API key
    )

def upload_file():
    # File Upload Widget
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "txt", "xlsx", "pdf"],
        help="Upload files in CSV, TXT, XLSX, or PDF formats"
    )
    return uploaded_file

class CodeResponse(BaseModel):
    code: str = Field(description="Pure code snippet without any additional text")

def extract_pure_code(code_message):
    # Extract content from AIMessage
    code_string = code_message.content if hasattr(code_message, 'content') else str(code_message)
    
    # Remove code block markers and clean up
    code_string = code_string.replace('```python', '').replace('```', '').strip()
    
    # Remove any leading comments or docstrings (single-line or multi-line)
    # Remove multiline docstrings (triple quotes)
    code_string = re.sub(r'"""(.*?)"""', '', code_string, flags=re.DOTALL)
    code_string = re.sub(r"'''(.*?)'''", '', code_string, flags=re.DOTALL)
    
    # Remove single-line comments (start with #)
    lines = code_string.split('\n')
    pure_code_lines = [line for line in lines if not line.strip().startswith('#')]

    # Join the remaining lines of pure code
    return '\n'.join(pure_code_lines).strip()
    
def generate_pure_code(programming_language: str, problem_description: str, df: pd.DataFrame, filename: str):
    df_preview = df.head(1).to_string()

    # Create a prompt template
    prompt = PromptTemplate(
        template=(
            "You are an expert Python programmer specializing in machine learning.\n"
            "Generate a concise, production-ready code solution in {language} for the following problem:\n"
            "{problem}\n\n"
            "Data Preview (first 1 rows):\n{df_preview}. wrie a code for file name {filename}.\n\n"
            "Requirements:\n"
            "1. Provide ONLY the code implementation\n"
            "2. Ensure the code is clean, efficient, and solves the problem directly\n"
            "3. Include necessary imports and comments for clarity\n"
            "4. Handle potential errors and edge cases\n"
            "5. Save the trained model"
            "6. Select model by analyzing the data. If it's a regression or classification."
        ),
        input_variables=["language", "problem", "df_preview"]
    )
    
    # llm = Ollama(
    #     model = "codegemma:2b"
    # )

    # Initialize ChatOpenAI with enhanced parameters
    
    try:
        # Generate response using the formatted prompt
        response = llm.invoke(
            prompt.format(
                language=programming_language, 
                problem=problem_description,
                df_preview=df_preview,
                filename = filename
            )
        )
        
        # Extract and clean pure code
        pure_code = extract_pure_code(response)
        
        return pure_code, response
    
    except Exception as e:
        print(f"Code generation error: {e}")
        return ""

def execute_generated_code(code: str):
    try:
        # Redirect stdout and stderr to capture execution output
        stdout = io.StringIO()
        stderr = io.StringIO()
        sys.stdout = stdout
        sys.stderr = stderr
        
        # Convert warnings to exceptions
        warnings.simplefilter("error", UserWarning)

        # Safe execution environment
        exec_globals = {"pd": __import__("pandas")}
        exec_locals = {}

        # Execute code
        exec(code, exec_globals, exec_locals)
        
        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Capture output
        execution_output = stdout.getvalue()
        return {"success": True, "output": execution_output, "error": None}
    
    except Exception as e:
        # Capture the traceback
        error_trace = traceback.format_exc()
        
        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        execution_error = stderr.getvalue() + error_trace
        return {"success": False, "output": None, "error": execution_error}

def solve_errors(code: str, error: str):
    st.write("### Analyzing and Fixing Errors")
    st.error(f"Error encountered:\n{error}")
    
    # Add specific checks for common errors
    if "unterminated string literal" in error:
        st.warning("Detected unterminated string. Attempting to fix...")
        # Add closing quotes to any unterminated strings
        lines = code.splitlines()
        fixed_lines = []
        for line in lines:
            if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                line += '"' if '"' in line else "'"
            fixed_lines.append(line)
        code = "\n".join(fixed_lines)
    
    elif "unexpected EOF while parsing" in error:
        st.warning("Detected incomplete syntax. Checking for missing brackets or quotes...")
        # Add logic to check for missing brackets or quotes (simplified example)
        if code.count("(") > code.count(")"):
            code += ")"
        elif code.count("{") > code.count("}"):
            code += "}"
    
    # Refine the code with the LLM
    prompt = PromptTemplate(
        template=(
            "You are an expert Python programmer. The following code contains an error:\n\n"
            "Code:\n{code}\n\n"
            "Error:\n{error}\n\n"
            "Fix the error and provide corrected code."
            "Requirements:\n"
            "1. Provide ONLY the code implementation and no other texts\n"
            "2. Ensure the code is clean, efficient, and solves the problem directly\n"
            "3. Include necessary imports and comments for clarity\n"
            "4. All the columns of the data should be cleaned for a Machine Learning model\n"
            "5. Identify data type and which ML model will suit best. Train the model and save trained Model with performance metrix."
        ),
        input_variables=["code", "error"]
    )
    
    try:
        response = llm.invoke(
            prompt.format(
                code=code,
                error=error
            )
        )
        return extract_pure_code(response)
    except Exception as e:
        st.error(f"Error while generating fixed code: {e}")
        return code  # Return the original code if fixing fails


def execute_and_fix_code(generated_code: str, max_retries: int = 3):
    """
    Execute the generated code and iteratively fix errors until successful or retry limit is reached.
    
    Args:
        generated_code (str): The initial code to execute.
        max_retries (int): The maximum number of retries to fix errors.
    
    Returns:
        dict: A dictionary containing the success status, output, and the final version of the code.
    """
    retry_count = 0
    while retry_count < max_retries:
        # Execute the code
        execution_result = execute_generated_code(generated_code)

        if execution_result["success"]:
            # Code executed successfully
            return {
                "success": True,
                "output": execution_result["output"],
                "final_code": generated_code,
                "error": None,
            }
        else:
            # Log the error details for debugging
            st.write(f"### Error during execution (Attempt {retry_count + 1}):")
            st.write(execution_result["error"])
            
            # Generate a corrected version of the code
            generated_code = solve_errors(generated_code, execution_result["error"])
            retry_count += 1

    # If all retries failed
    return {
        "success": False,
        "output": None,
        "final_code": generated_code,
        "error": f"Failed after {max_retries} attempts. Last error: {execution_result['error']}",
    }

def main():
    st.title("📁 File Upload Utility")

    uploaded_file = upload_file()

    df = None

    # File Processing
    if uploaded_file is not None:
        # Get file details
        file_details = {
            "Filename": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size": f"{uploaded_file.size} bytes"
        }
        st.write("### File Details")
        st.write(file_details)

        # Process different file types
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.write("### CSV Preview")
            st.dataframe(df)
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            st.write("### Excel Preview")
            st.dataframe(df)
        
        else:
            # For other file types, show raw content
            st.write("### File Content")
            st.write(uploaded_file.getvalue())

    if df is not None:
        problems = [
            "perform data wrangling to train machine learning model on the given data. Keep the file name cleaned_data_1.csv."
        ]

        # Generate and execute code for each problem
        for problem in problems:
            st.write(f"### Generating code for: {problem}")
            
            # Generate code
            generated_code, full_code = generate_pure_code(
                programming_language="Python", 
                problem_description=problem, 
                df=df,
                filename=uploaded_file.name
            )
            
            # Execute and fix the generated code
            result = execute_and_fix_code(generated_code, max_retries=5)

            if result["success"]:
                st.write("### Execution Successful!")
                st.write("Generated Code:")
                st.code(result["final_code"], language="python")
                
                # Reload and display the cleaned data
                cleaned_df = pd.read_csv("cleaned_data_1.csv")
                st.write("### Output CSV Preview")
                st.dataframe(cleaned_df)
            else:
                st.write("### Execution Failed")
                st.write("Final Generated Code:")
                st.code(result["final_code"], language="python")
                st.write("Error Details:")
                st.write(result["error"])

if __name__ == "__main__":
    main()