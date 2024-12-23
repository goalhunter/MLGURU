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
from dotenv import load_dotenv
import os
from io import BytesIO
import matplotlib.pyplot as plt

load_dotenv()
llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,  # Creativity level
        api_key=os.getenv('OPENAI_API_KEY')  # Replace with your actual API key
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
 
def generate_pure_code(programming_language: str, problem_description: str, df: pd.DataFrame):
    df_preview = df.head(5).to_string()
    column_info = df.dtypes.to_string()

    prompt = PromptTemplate(
        template=(
            "You are an expert Python programmer specializing in data analysis, cleaning, and machine learning.\n"
            "Generate production-ready code in {language} for the following user request:\n"
            "{problem}\n\n"
            "Data Preview:\n{df_preview}\n\n"
            "Column Types:\n{column_info}\n\n"
            "file is already uploaded on streamlit and extracted the file data as df.\n"
            "Requirements:\n"
            "1. Address the user's specific request while ensuring data quality:\n"
            "   - Handle missing values appropriately\n"
            "   - Fix data types as needed\n"
            "   - Process data according to the user's requirements\n"
            "   - Apply appropriate visualizations if requested\n"
            "2. Include all necessary imports\n"
            "3. Use descriptive variable names\n"
            "4. Return the processed DataFrame as 'result_df'\n"
            "5. Include relevant progress messages using st.write()\n"
            "6. If analysis or ML is requested, include appropriate metrics and evaluations\n"
            "7. Only give the code and no other texts"
        ),
        input_variables=["language", "problem", "df_preview", "column_info"]
    )

    try:
        response = llm.invoke(
            prompt.format(
                language=programming_language,
                problem=problem_description,
                df_preview=df_preview,
                column_info=column_info
            )
        )
        return extract_pure_code(response), response
    except Exception as e:
        print(f"Code generation error: {e}")
        return "", None

def execute_generated_code(code: str, df: pd.DataFrame):
    try:
        # Redirect stdout and stderr to capture execution output
        stdout = io.StringIO()
        stderr = io.StringIO()
        sys.stdout = stdout
        sys.stderr = stderr

        # Safe execution environment with preloaded imports
        exec_globals = {
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "sklearn": __import__("sklearn"),
            "df": df,
        }
        exec_locals = {}

        # Execute the code
        exec(code, exec_globals, exec_locals)

        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Capture the modified DataFrame (assuming it's returned as `result_df`)
        result_df = exec_locals.get("result_df", None)
        execution_output = stdout.getvalue()

        return {"success": True, "output": execution_output, "result_df": result_df, "error": None}

    except Exception as e:
        # Capture the traceback
        error_trace = traceback.format_exc()

        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        execution_error = stderr.getvalue() + error_trace
        return {"success": False, "output": None, "result_df": None, "error": execution_error}


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

def execute_and_display_code(df: pd.DataFrame, generated_code: str, max_retries: int = 3):
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            st.write(f"Attempt {retry_count + 1} of {max_retries}")
            
            # Clear any existing matplotlib plots
            plt.close('all')
            
            # Create a safe execution environment
            local_vars = {
                "pd": pd,
                "np": __import__("numpy"),
                "sklearn": __import__("sklearn"),
                "df": df.copy(),
                "st": st,
                "plt": plt,
                "BytesIO": BytesIO
            }
            
            # Execute the code
            exec(generated_code, local_vars)
            
            # Check if result_df was created
            if "result_df" in local_vars:
                result_df = local_vars["result_df"]
                
                # Verify data quality
                if result_df is None or result_df.empty:
                    raise ValueError("Resulting DataFrame is empty or None")
                
                # Capture all current matplotlib figures
                figures = []
                for fig_num in plt.get_fignums():
                    figures.append(plt.figure(fig_num))
                
                # Check for saved PNG files in the local variables
                png_files = {k: v for k, v in local_vars.items() if isinstance(v, str) and v.endswith('.png')}
                
                return {
                    "success": True,
                    "result_df": result_df,
                    "error": None,
                    "figures": figures,
                    "png_files": png_files
                }
            else:
                raise ValueError("No result_df was created")
                
        except Exception as e:
            last_error = str(e)
            st.write(f"Error in attempt {retry_count + 1}: {last_error}")
            
            # Generate fixed code
            prompt = PromptTemplate(
                template=(
                    "Fix the following Python code that produced an error:\n"
                    "Code:\n{code}\n\n"
                    "Error:\n{error}\n\n"
                    "Data is already uplaoded in streamlit in a variable called df \n"
                    "Requirements:\n"
                    "1. Fix the error while maintaining the user's requested functionality\n"
                    "2. Return the processed DataFrame as 'result_df'\n"
                    "3. For visualizations, either use plt.show() or save to 'result.png'\n"
                    "4. Ensure robust error handling\n"
                    "5. Only write the code and no other texts and explainations"
                ),
                input_variables=["code", "error"]
            )
            
            try:
                response = llm.invoke(
                    prompt.format(code=generated_code, error=last_error)
                )
                generated_code = extract_pure_code(response)
            except Exception as gen_error:
                st.error(f"Error generating fixed code: {gen_error}")
            
            retry_count += 1
    
    return {
        "success": False,
        "result_df": None,
        "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
        "figures": [],
        "png_files": {}
    }

def main():
    st.title("🤖 AI-Powered Data Processing Assistant")
    
    # Add description
    st.markdown("""
    Upload your data and describe what you want to do with it. Examples:
    - "Clean the data and handle missing values"
    - "Create a visualization showing the relationship between price and size"
    - "Train a model to predict the target column"
    - "Perform exploratory data analysis"
    """)
    
    uploaded_file = upload_file()
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            file_type = uploaded_file.type
            if file_type == "text/csv":
                df = pd.read_csv(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type")
                return
            
            # Data Overview Section
            st.write("### 📊 Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Preview:")
                st.dataframe(df.head())
            with col2:
                st.write("Dataset Info:")
                st.write(f"Rows: {df.shape[0]}")
                st.write(f"Columns: {df.shape[1]}")
                st.write("Column Types:")
                st.write(df.dtypes)
            
            # User Input Section
            st.write("### 🎯 What would you like to do with your data?")
            
            example_tasks = [
                "Select an example task",
                "Clean the data and handle missing values",
                "Perform exploratory data analysis",
                "Create visualizations of key relationships",
                "Train a machine learning model",
                "Calculate summary statistics",
                "Custom task (specify below)"
            ]
            selected_task = st.selectbox("Choose a task or select 'Custom task'", example_tasks)
            
            if selected_task == "Custom task (specify below)":
                user_input = st.text_area(
                    "Describe what you want to do with your data:",
                    height=100,
                    placeholder="E.g., Create a scatter plot comparing sales and marketing spend, then calculate correlation"
                )
            else:
                user_input = selected_task if selected_task != "Select an example task" else ""
            
            if user_input and st.button("🚀 Process Data"):
                st.write("### 🔄 Generating and Executing Code")
                generated_code, _ = generate_pure_code("Python", user_input, df)
                
                if generated_code:
                    with st.expander("View Generated Code"):
                        st.code(generated_code, language="python")
                    
                    result = execute_and_display_code(df, generated_code, max_retries=5)
                    
                    if result["success"]:
                        st.success("✅ Processing completed successfully!")
                        st.code(generated_code, language="python")
                        processed_df = result["result_df"]
                        st.write("### 📈 Results")
                        st.dataframe(processed_df.head())
                        
                        # Provide download option
                        csv = processed_df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Processed Data",
                            csv,
                            "processed_data.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                        # Display visualizations
                        if result["figures"] or result["png_files"]:
                            st.write("### 📊 Visualizations")
                            
                            # Display matplotlib figures
                            for i, fig in enumerate(result["figures"]):
                                st.pyplot(fig)
                            
                            # Display saved PNG files
                            for filename in result["png_files"].values():
                                try:
                                    image = plt.imread(filename)
                                    st.image(image, caption=f"Generated visualization: {filename}")
                                except Exception as e:
                                    st.error(f"Error displaying image {filename}: {e}")
                            
                    else:
                        st.error("❌ Processing failed")
                        st.error(result["error"])
                else:
                    st.error("Failed to generate processing code")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()