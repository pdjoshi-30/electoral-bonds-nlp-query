# Electrol Bonds nlp Query
 This project is the comprehensive implementation of RAG application of PDF and CSV data which have large pages.
 This project is done as a part of 1 Day Hackathon conducted by Microsoft sponsored ACM Summer School, IIT Gandhinagar we were assign to do this project on Elecctrol Bonds but this can be extended to any other pdfs


The project is implemented using langchai, FAISS and the concepts of LLMs. The code is a slow implementation Because the vector embeddings and the Phi's index takes a lot of time to compute And that is why we have created another notebook for faster implementation, which uses the smart document data frame from Pandas Ai Library which computes the embedding faster so that we can make an inference.

There is a still lot's of Scope in this Project and We are working on this ,feel free to give any Suggestions that will help us to improve the project/


To setup this project, you need to run the following commands:

```bash
pip install -r requirements.txt
```

To run this project, you need to run the following command:

```bash
streamlit run app.py
```
The faster inference code use PANDASAI and GROQ library to compute the embeddings faster. The code is implemented in the app.py under the comments.
