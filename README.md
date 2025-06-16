# NLP-Exam

Semantic reconstruction is a crucial task in Natural Language Processing , aiming to reform a text while preserving its original meaning . Semantic reconstruction is really usefull for getting a better quality and understanding from the original text , that could have writting problems . This , can be automated , extendable and adjustable for its occasion , so it can be used in education , healthcare , business etc.

<!-- Expanation for the first assigment -->

For Question A, we were asked to paraphrase a sentence of our choice using a custom NLP approach. To achieve this, we first created a function called get_synonym() which takes a word as input and returns the first synonym it finds from a list of synonyms.
Next, we implemented a function named replace_adjectives(). This function takes a sentence as input and uses the spaCy library to tokenize it. For each token (word), it checks whether the token is an adjective. If it is, the function calls get_synonym() to retrieve a synonym and replaces the adjective with it. All other tokens remain unchanged.
Finally, we defined one or more example sentences and applied the replace_adjectives() function to them in order to generate their paraphrased versions, where adjectives were substituted with their synonyms .
For question B , we were asked to paraphrase the whole texts with the additional help of hugging face pipelines . So , we used three different text to text generation pipelines . The first one is "tuner007/pegasus_paraphrase" and for this one , we split the text into sentences , we paraphrased each sentence and the we added then in a new string . For the next 2 pipelines which are "google/flan-t5-large" , "eugenesiow/bart-paraphrase", we used the same method which is to paraphrase the text at once .
For question C , we asked openAI to generate as the paraphrase for the two texts. We supposed that it was the perfect paraphrase it could be and we were asked to compare it with the paraphrases our pipilines did . The library evaluate helps as import bertscore , which compers semanticly the the paraphrased text and the original . So , we made a loop for every paraphrased text to compare it with the original and gives the final score for each comparison . The score has to be smaller than one and the closer to one it is , the best score it has , that means that the paraphrase doesn't lose the point

<!-- Expanation for the second assigment -->

For the second assignment, we were asked to use word embeddings to compare the vocabulary between reconstructed texts and their original versions, and to visualize the results. To achieve this, we implemented the function preprocess_and_embed(sentence, model, stop_words, stemmer), using the NLTK, NumPy, Gensim, and scikit-learn libraries. This function tokenizes the sentence, removes stopwords and punctuation, and applies stemming to the remaining words. For each stemmed word, we attempt to retrieve its embedding using a pre-trained model from Gensim. If no embedding is found, we reverse the stemming to try the original word instead.
Next, we created the function full_similarity_matrix(embed1, embed2), which compares two sets of word embeddings. For each word in embed1, it calculates cosine similarities against all words in embed2 and stores the most semantically similar match along with its score. To evaluate the overall similarity between the two texts, we implemented average_cosine_similarity(embed1, embed2), which computes the average of these best cosine similarity scores.
Finally, we visualized the semantic relationships using the function visualize_embeddings(embeddings1, embeddings2, method='pca', title=''). This function merges the two embedding sets and applies either PCA or t-SNE to reduce them to two dimensions. The resulting coordinates are plotted, with blue dots representing words from the original sentence and red dots for the reconstructed one. Each word is labeled with the same colors . This visualization helps us understand whether the semantic space of the reconstructed text closely aligns with the original.

<!-- How significant were words embeddings for the project -->

Word embeddings played a crucial role in this project. By utilizing pretrained models such as Word2Vec and GloVe, we were able to compute vector representations of words and evaluate cosine similarity between original and reconstructed sentences. In cases where the paraphrased text preserved similar vocabulary and grammatical structure, the cosine similarity scores were consistently high (typically above 0.7), indicating successful semantic preservation.
However, there were instances where the models failed to fully capture deeper semantic meanings . In such cases, the similarity scores were lower, reflecting a loss in semantic alignment.

<!-- What were the biggest challenges for this project -->

Significant challenges during the process included finding appropriate synonyms that preserved the original meaning, handling mismatches in word embeddings caused by stemming, and maintaining syntax in the reconstructed sentences. Finding appropriate synonyms was difficult because not always the synonys matched the sentence meaning because it was slightly different . Stemming created a missmatching for the embeddings , which we handled when that happened to reconstruct the full word and search again . Sometimes , when the words were changed without grammatic caustion , the syntax was getting lost .

<!-- How can it be automated  -->

The paraphrasing can be automated by using the scripts we created with a spacific way . First paraphrase the whole text , token the words , remove stopwords , calculate embedings and cosine similarity and opticalise it . Finaly , the user inserts a text and then by the correct method, the job is done .

<!-- Comparison between pipelines and custom nlp -->

Every pipeline is different between each other because is trained with different data . Also , it is created with different methods , so every pipeline handles the text or the sentences with a different way . As for the custom nlp , it completly different the way it paraphrases the text because it is not trained and it only rephrase the adjectives of the sentence with the synonyms , so the work is really amateur and not fulfilled as the pretrained pipeline models .

<!-- Conclusion -->

NLP is very usefull for the developers , because we can get the job done with automated scripts that are "clever" and built by our spacific needs . Artificial "intelligence" can be imported by NLP because the computer can't understand words but it can understand numbers , so we make the words , numbers , as embeddings and then the computer with the pretrained models does the work tha we need and we reach our final assigment .

<!-- Bibliography -->

For this project we used knowledge we searched at :
The presentation of the class "Επεξεργασία στη Φυσική γλώσσα" ,
The libraries tha we used for the project ,
https://huggingface.co/docs/transformers/en/main_classes/pipelines#natural-language-processing ,
https://stackoverflow.com/questions/46433778/import-googlenews-vectors-negative300-bin
