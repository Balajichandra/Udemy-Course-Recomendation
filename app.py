from flask import Flask,request,render_template
from os import read
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def readdata():
    df = pd.read_csv('UdemyCleanedTitle.csv')
    return df

def getcleantitle(df):
    df['Clean_title'] = df['course_title'].apply(nfx.remove_stopwords)
    df['Clean_title'] = df['course_title'].apply(nfx.remove_special_characters)
    return df

def getcosinemat(df):
    countvect = CountVectorizer()
    cvmat = countvect.fit_transform(df)
    return cvmat

def cosinemmat(cvmat):
    return cosine_similarity(cvmat)

def recomended_course(df,titlename,cosine_mat,num_rec):
    course_index = pd.Series(df.index,index=df['course_title']).drop_duplicates()
    index = course_index['title']
    scores = list(enumerate(cosine_mat[index]))
    sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
    selected_course_index = [i[0] for i in sorted_scores[1:]]
    selected_course_score = [i[1] for i in sorted_scores[1:]]
    rec_df = df.iloc[selected_course_index]
    rec_df['Similarity_Score'] = selected_course_score
    final_recommended_courses = rec_df[['course_title', 'Similarity_Score', 'url', 'price', 'num_subscribers']]
    return final_recommended_courses.head(num_rec)

def extractfeature(recdf):
    course_url = list(recdf['url'])
    course_title = list(recdf['course_title'])
    course_price = list(recdf['price'])
    return course_url,course_title,course_price

def searchterm(titlename,df):
    result_df = df[df['course_title'].str.contains(titlename)]
    top6 = result_df.sort_values(by="num_subscribers",ascending=False).head(6)
    return top6

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':
        my_dict = request.form 
        titlename = my_dict['course']

        try:
            df = readdata()
            df = getcleantitle(df)
            cvmat = getcosinemat(df)
            num_rec = 6
            cosine_mat = cosinemmat(cvmat)
            recdf = recomended_course(df,titlename,cosine_mat,num_rec)
            course_url,course_title,course_price = extractfeature(recdf)
            dictmap = dict(zip(course_title,course_url))
            if len(dictmap) != 0:
                return render_template('index.html',coursemap = dictmap,coursename = titlename,showtitle = True)
            else:
                return render_template('index.html',showerror=True,coursename=titlename)

        except:
            result_df = searchterm(titlename,df)
            if result_df.shape[0] > 6:
                result_df = result_df.head(6)
                course_url,course_title,course_price = extractfeature(result_df)
                coursemap = dict(zip(course_title, course_url))
                if len(coursemap) != 0:
                    return render_template('index.html', coursemap=coursemap, coursename=titlename, showtitle=True)

                else:
                    return render_template('index.html', showerror=True, coursename=titlename)
            else:
                course_url, course_title, course_price = extractfeature(result_df)
                coursemap = dict(zip(course_title, course_url))
                if len(coursemap) != 0:
                    return render_template('index.html', coursemap=coursemap, coursename=titlename, showtitle=True)

                else:
                    return render_template('index.html', showerror=True, coursename=titlename)    
        return render_template('index.html')        
    if __name__ == '__main__':
        app.run(debug=True)