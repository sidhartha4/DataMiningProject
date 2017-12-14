from sklearn.feature_extraction.text import TfidfVectorizer



def main():

	corpus = ["This is very strange",
	          "This is very nice"]


	Vocabulary = {"is":0, "strange":1, "nice":2}


	vectorizer = TfidfVectorizer(min_df=1, vocabulary=Vocabulary)


	X = vectorizer.fit_transform(corpus).toarray()

	print(X)


	corpus = ["This is very strange",
	          "This is very nice"]

	from sklearn.feature_extraction.text import CountVectorizer
	vec = CountVectorizer(vocabulary=Vocabulary)
	data = vec.fit_transform(corpus).toarray()
	print(data)


if __name__ == '__main__':
	main()