import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Loan Dataset Viewer")

# File uploader widget
uploaded_file = st.file_uploader("Upload your loan_data_set.csv file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())  # Display the first few rows of the dataset
        
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Filter data for Applicant Income <= 40000
        df = df[df['ApplicantIncome'] <= 40000] # Properly aligned with parent block

        st.write("### Filtered Dataset (Applicant Income ≤ 40000)")
        st.write(df.head())  # Display the first few rows of the filtered dataset
except Exception as e:
        st.error(f"An error occured: {e}")

# Plot histogram
        st.write("### Distribution of Applicant Income (Up to 40000)")
        fig, ax = plt.subplots()
        df['ApplicantIncome'].hist(bins=50, ax=ax)
        ax.set_xlabel('Applicant Income')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Applicant Income (up to 40000)')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Coapplicant Income Analysis
        st.write("## Coapplicant Income Analysis (Up to 20000)")
        df_coapplicant = df[df['CoapplicantIncome'] <= 20000]
        st.write("Filtered Dataset (Coapplicant Income ≤ 20000):")
        st.write(df_coapplicant.head())

        # Plot Coapplicant Income Distribution
        fig2, ax2 = plt.subplots()
        df_coapplicant['CoapplicantIncome'].hist(bins=50, ax=ax2)
        ax2.set_xlabel('Coapplicant Income')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Coapplicant Income (Up to 20000)')
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Loan Amount Analysis
        st.write("## Loan Amount Analysis")
        fig3, ax3 = plt.subplots()
        df['LoanAmount'].hist(bins=50, ax=ax3)
        ax3.set_xlabel('Loan Amount')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Loan Amount')
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Gender Distribution
        st.write("## Gender Distribution")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Gender', data=df, ax=ax4)
        ax4.set_title('Gender Distribution')
        ax4.set_xlabel('Gender')
        ax4.set_ylabel('Count')
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Married Distribution
        st.write("## Married Distribution")
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Married', data=df, ax=ax5)
        ax5.set_title('Married Distribution')
        ax5.set_xlabel('Married')
        ax5.set_ylabel('Count')
        st.pyplot(fig5)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Dependents Distribution
        st.write("## Dependents Distribution")
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Dependents', data=df, ax=ax6)
        ax6.set_title('Dependents Distribution')
        ax6.set_xlabel('Number of Dependents')
        ax6.set_ylabel('Count')
        st.pyplot(fig6)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Education Distribution
        st.write("## Education Distribution")
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Education', data=df, ax=ax7)
        ax7.set_title('Education Distribution')
        ax7.set_xlabel('Education Level')
        ax7.set_ylabel('Count')
        st.pyplot(fig7)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
