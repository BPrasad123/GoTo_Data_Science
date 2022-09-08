## tf-idf ##
Explanation and code in detail:
https://www.kdnuggets.com/2022/09/convert-text-documents-tfidf-matrix-tfidfvectorizer.html

Apply tf-idf to each column separately instead of concatenation of all the vocabularies in the dataframe:  
Uses make_column_transformer from sklearn compose to apply tf--idf instance across columns independently.  
[Post](https://www.linkedin.com/feed/update/urn:li:activity:6973297572564140033/?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A6973297572564140033%29) and [Code](https://colab.research.google.com/drive/1Tq55a6iXCRFVis_pt3Iy-XDER7-YYQba?usp=sharing) by Dipanjan Sarkar


## key words, key phrases, topic modeling ##
1. Key word or key phrase extraction methods like RAKE or YAKE
2. Key word extraction methods with transformer embeddings like KeyBERT
3. Topic models like LDA and BERTopic which uses transformers and c-TF-IDF

Unsupervised topic modeling: LDA Rake, Yake

[Post with good discussion](https://www.linkedin.com/feed/update/urn:li:activity:6972934394822950912/?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A6972934394822950912%29)

