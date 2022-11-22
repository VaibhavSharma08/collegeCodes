import flask
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

app = flask.Flask(__name__, template_folder='templates')

df2 = pd.read_csv('./model/tmdb.csv')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
# print(count_matrix.get_shape())
# print(type(count_matrix[0]))
# print(count_matrix[0])
# print(count_matrix.toarray())


def recommenderSystem(X, Y, i):
    X = X.toarray()
    Y = Y.toarray()
    output = []
    # Option 1 - Make list beforehand and then just fetch for each recommendation
    # for i in range(len(X)):
    #     output.append([])
    #     for j in range(len(Y)):
    #         num = sum(x == y for x, y in zip(X[i], Y[j]))
    #         output[-1].append(num)
    #         print(num)
    # print(len(output))
    # print(len(output[0]))

    # Option 2 - Calculate result seperately for each recommendation
    for j in range(len(Y)):
        num = sum((x == y and x != 0) for x, y in zip(X[i], Y[j]))
        output.append(num)
        print("{} : {}".format(j, num))
    print(len(output))
    return output


def recommenderSystemOptimized(X, Y, i):
    X = X.toarray()
    Y = Y.toarray()
    output = []

    # Option 1 - Make list beforehand and then just fetch for each recommendation
    for i in range(len(X)):
        output.append([])
        for j in range(len(Y)):
            num = np.dot(X[i], Y[j])
            output[-1].append(num)
            print("{} : {}".format(j, num))
    # print(len(output))
    # print(len(output[0]))

    # Option 2 - Calculate result seperately for each recommendation
    # for j in range(len(Y)):
    #     num = np.dot(X[i], Y[j])
    #     output.append(num)
    #     print("{} : {}".format(j, num))
    # print(len(output))
    return output


def recommenderSystemSuperOptimized(X, Y, i):
    X = X.toarray()
    Y = Y.toarray()
    output = []

    # Option 1 - Make list beforehand and then just fetch for each recommendation
    for i in range(len(X)):
        num = X[i] @ np.transpose(Y)
        output.append(num)
        print("{} : {}".format(i, num))
    print(len(output))
    print(len(output[0]))

    # Option 2 - Calculate result separately for each recommendation
    # output = X[i] @ np.transpose(Y)
    # print("{}".format(output))
    # print(len(output))
    return output


def recommenderSystemSuperSuperOptimized(X, Y):
    # Option 1 - Make list beforehand and then just fetch for each recommendation
    output = safe_sparse_dot(X, Y.T, dense_output=True)

    return output


# cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# print(len(cosine_sim2))
# print(len(cosine_sim2[0]))
# print(cosine_sim2)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
all_titles = [df2['title'][i] for i in range(len(df2['title']))]


def get_recommendations(title):
    idx = indices[title]

    # output = cosine_similarity(count_matrix, count_matrix)
    # output = recommenderSystem(count_matrix, count_matrix, idx)
    # output = recommenderSystemOptimized(count_matrix, count_matrix, idx)
    # output = recommenderSystemSuperOptimized(count_matrix, count_matrix, idx)
    output = recommenderSystemSuperSuperOptimized(count_matrix, count_matrix)

    sim_scores = list(enumerate(output[idx]))      # For Option 1
    # sim_scores = list(enumerate(output))    # For Option 2

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = df2['title'].iloc[movie_indices]
    dat = df2['release_date'].iloc[movie_indices]
    gen = df2['genres'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title', 'Year', 'Genres'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return_df['Genres'] = gen
    return return_df


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
        check = difflib.get_close_matches(m_name, all_titles, cutoff=0.50, n=1)
        m_name = check[0]
        if m_name not in all_titles:
            return (flask.render_template('negative.html', name=m_name))
        else:
            result_final = get_recommendations(m_name)
            names = []
            dates = []
            genres = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])
                genres.append(result_final.iloc[i][2])

            return flask.render_template('positive.html', movie_names=names, movie_date=dates, movie_genre=genres, search_name=m_name)


if __name__ == '__main__':
    app.run()
