from django.shortcuts import render
import pickle

def home(request):
    return render(request, 'index.html')

def getPredictions(genre, platform, release_year, developer, players, peak_players, metacritic, esports, trending):
    model = pickle.load(open("d:/lol/GamingPrediction/thegame/base/ml_model.sav", "rb"))
    scaler = pickle.load(open("d:/lol/GamingPrediction/thegame/base/scaler.sav", "rb"))
    
    # Convert categorical variables to numeric
    genre_map = {'Action': 0, 'Adventure': 1, 'RPG': 2, 'Strategy': 3}
    platform_map = {'PC': 0, 'PlayStation': 1, 'Xbox': 2, 'Nintendo': 3}
    trending_map = {'Trending': 2, 'Stable': 1, 'Declining': 0}
    
    genre_num = genre_map[genre]
    platform_num = platform_map[platform]
    trending_num = trending_map[trending]
    
    # Print values for debugging
    print(f"Input values: {genre_num}, {platform_num}, {release_year}, {players}, {peak_players}, {metacritic}, {esports}, {trending_num}")
    
    features = [genre_num, platform_num, int(release_year), float(players), 
               int(peak_players), float(metacritic), float(esports), trending_num]
    
    prediction = model.predict(scaler.transform([features]))
    
    # Print prediction for debugging
    print(f"Raw prediction: {prediction[0]}")
    
    return f'{prediction[0]:.2f}'

def result(request):
    game_title = request.GET['game_title']
    genre = request.GET['genre']
    platform = request.GET['platform']
    release_year = request.GET['release_year']
    developer = request.GET['developer']
    players = request.GET['players']
    peak_players = request.GET['peak_players']
    metacritic = request.GET['metacritic']
    esports = request.GET['esports']
    trending = request.GET['trending']
    
    result = getPredictions(genre, platform, release_year, developer, players, peak_players, metacritic, esports, trending)
    
    return render(request, 'result.html', {'result': result})
