from django.shortcuts import render
import pickle

def home(request):
    return render(request, 'index.html')

def getPredictions(genre, platform, release_year, players, peak_players, metacritic, trending):
    model = pickle.load(open("d:/lol/GamingPrediction/Model/ml_model.sav", "rb"))
    scaler = pickle.load(open("d:/lol/GamingPrediction/Model/scaler.sav", "rb"))
    
    # Convert categorical variables
    genre_map = {
        'Action': 0, 'Adventure': 1, 'RPG': 2, 'Strategy': 3, 'Sports': 4,
        'Racing': 5, 'Horror': 6, 'Fighting': 7, 'Shooter': 8, 'Simulation': 9
    }
    platform_map = {
        'PC': 0, 'PlayStation': 1, 'Xbox': 2, 'Nintendo Switch': 3,
        'Mobile': 4, 'Cross-Platform': 5
    }
    trending_map = {'Rising': 2, 'Stable': 1, 'Declining': 0}
    
    # Calculate features with adjusted scale
    player_retention = (float(peak_players) / (float(players) * 1000000)) * 100  # Convert to percentage
    recent_release = min(2024 - int(release_year), 5)
    high_rating = 1 if float(metacritic) >= 75 else 0
    
    features = [
        genre_map[genre],
        platform_map[platform],
        recent_release,
        player_retention,
        high_rating,
        trending_map[trending]
    ]
    
    prediction = model.predict(scaler.transform([features]))
    return "Yes" if prediction[0] == 1 else "No"

def result(request):
    genre = request.GET['genre']
    platform = request.GET['platform']
    release_year = request.GET['release_year']
    players = request.GET['players']
    peak_players = request.GET['peak_players']
    metacritic = request.GET['metacritic']
    trending = request.GET['trending']
    
    result = getPredictions(genre, platform, release_year, players, peak_players, metacritic, trending)
    return render(request, 'result.html', {'result': result})
