import io
import os
import re
import time
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import xgboost as xgb
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras.optimizers import SGD
from sklearn import preprocessing

def make_spotify():
	cid = '1db39bea540c44b28de5f4945e31c8fb'
	secret = '1208087c427c4128a98f3c4fade7a60a'
	client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
	return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def train_old_new_algo():

	if "model_data_wgenre.csv" not in os.listdir():
		raise Exception("Could not find training data, shutting down, goodbye :(")

	print("Training old/new model now beep boop . . .")
	raw_train = pd.read_csv("model_data_wgenre.csv").drop_duplicates()
	old = raw_train[raw_train.is_old == 1]
	new = raw_train[raw_train.ryear >= 2000]
	training = pd.concat([old, new])
	X = training[["danceability", "energy", "loudness", "mode", "acousticness", "valence", "tempo", "duration_ms"]]
	Y = training["is_old"]

	boosted_model =xgb.XGBClassifier(n_estimators = 500, random_state=1,learning_rate=0.01, max_depth = 10)
	boosted_model.fit(X, Y)
	time.sleep(3)
	print("Old/new model trained!")
	return (training, boosted_model)

def train_genre_algo():

	if "model_data_wgenre.csv" not in os.listdir():
		raise Exception("Could not find training data, shutting down, goodbye :(")
		
	print("Training genre model now beep boop . . .")
	raw_train = pd.read_csv("model_data_wgenre.csv").drop_duplicates()
	training = raw_train[raw_train.genre != "unknown"]

	X = training[["danceability", "energy", "loudness", "mode", "acousticness", "valence", "tempo", "duration_ms"]]
	lab_dict = {}
	for num, label in enumerate(pd.Series(X.genre.unique()).sort_values()):
  		lab_dict[label] = num
  	back_to_label = {y:x for x,y in lab_dict.items()}
	Y = training["genre"].map{lab_dict}

	boosted_model = xgb.XGBClassifier(n_estimators = 1000, random_state=1,learning_rate=0.01, max_depth = 5, objective='multi:softmax', num_classes=8)
	boosted_model.fit(X, Y)
	print("Genre model trained!")
	return (back_to_label, boosted_model)


def get_genre(sp, songid):
	genre_list = sp.artist(sp.track(songid)["artists"][0]["id"])['genres']
	if len(genre_list) == 0:
		return "unknown"
	genre_sentence = " ".join(genre_list)
	rezlist = []
	for each in ['hip hop', 'rock', 'soul', 'country', 'pop', 'jazz', 'edm', 'reggae', 'classical']:
		mytup = (each, len(re.findall(each, genre_sentence)))
		rezlist.append(mytup)
	sorter = sorted(rezlist, key=lambda tup: tup[1], reverse = True)
	if sorter[0][1]==0:
		return "unknown"
	return sorter[0][0]

def get_song_info(sp, mysong, myartist, guess):

	songdict = sp.search(q='artist:' + myartist + ' track:' + mysong, type='track')["tracks"]["items"][guess]
	myid = songdict["id"]
	rdate = songdict["album"]["release_date"]
	artist_name = songdict["artists"][0]["name"]
	song_name = songdict["name"]
	return {"id":[myid], "artist_name": [artist_name], "song_name": [song_name], "rdate":[rdate]}

def check_song_name_ret_info(sp):
	song = input("Which song do you want me to guess? ")
	artist = input("Which artist sings it? ")

	ok = False
	guess = 0
	while ok == False:
		try:
			songinfo = pd.DataFrame(get_song_info(sp, song, artist, guess))
		except:
			print("Song lookup returned no results, try again")
			return (False, "blah")
		print("Is the song you want {s} by {a}?".format(s=songinfo["song_name"].iloc[0], a = songinfo["artist_name"].iloc[0]))
		good = input("Enter y or n ")
		if good[0] == "y":
			return (True, songinfo)
		else:
			ok= False
			guess += 1
			if guess >= 3:
				print("Can't find this song, please try another")
				return (False, songinfo)

def song_guesser(sp, modeltup):
	model = modeltup[1]

	
	# good = False
	# guess = 0
	# while good == False:
	# 	songtup = check_song_name_ret_info(sp, guess)
	# 	songinfo = songtup[1]
	# 	good = songtup[0]
	# 	guess += 1
	# 	if guess > 2:
	# 		print("Can't find this song, please try another")
	# 		guess = 0
	good = False

	while good == False:
		songtup = check_song_name_ret_info(sp)
		songinfo = songtup[1]
		good = songtup[0]

	print("Retrieving song features . . .")
	features = pd.DataFrame(sp.audio_features(songinfo["id"])[0], index=[0])

	song_model_data = songinfo.merge(features, how = 'inner', on = 'id')
	song_model_data["ryear"] = song_model_data.rdate.str.extract(r"(\d\d\d\d)").astype(int)
	song_model_data["is_old"] = np.where(song_model_data["ryear"] < 1980, 1, 0)

	print("Ok I'm ready to guess")
	time.sleep(3)
	print("I think your song is . .. ")
	time.sleep(3)
	deploy = song_model_data[["danceability", "energy", "loudness", "mode", "acousticness", "valence", "tempo", "duration_ms"]]
	myguess = model.predict(deploy)[0]
	if myguess == 1:
		print("OLD!")
	elif myguess == 0:
		print("NEW!")

	correct = input("Was I correct? Enter y or n ")

	if correct[0] == "y":
		print("Amazing! Thanks for playing!")
	else:
		print("Aw shucks! Let me learn from my mistake")
		print("Adding song to my training data")
		newtraining = pd.concat([modeltup[0],song_model_data])
		newtraining.to_csv("./model_data.csv", index = False)

	again = input("Wanna try another song? Enter y or n ")

	if again[0] == "y":
		return True
	else:
		print("Goodbye!")
		return False
		

def run_app():
	print("Welcome to Jared's Song Analyzer!\nMy name is SongBot and I will guess if a song is old or new.")
	print("I defined 'old' songs as released before 1980, and 'new' songs as released in the 2000's")
	print("First let me review some songs before I let you guess")
	input("Press Enter to train SongBot")
	modeltup = train_old_new_algo()
	sp = make_spotify()

	keep_playing = True

	while keep_playing == True:
		keep_playing = song_guesser(sp, modeltup)

def make_genre_table()
	sp = make_spotify()
	raw_train = pd.read_csv("model_data.csv").drop_duplicates()
	genre_list = []
	for each in raw_train.id:
		genre_list.append(get_genre(sp, each))
	raw_train['genre'] = genre_list
	raw_train.to_csv("model_data_wgenre.csv", index = False)


if __name__ == "__main__":
	run_app()



