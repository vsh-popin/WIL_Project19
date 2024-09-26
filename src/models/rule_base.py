from experta import *

class WeatherRecommender(KnowledgeEngine):

    @Rule(
        Fact(t_max=P(lambda x: 15 <= x < 20)),
        # Fact(t_min=P(lambda x: x > 15)),
        # Fact(wind_gust=P(lambda x: x == 15)),
        # Fact(precip=P(lambda x: x == 2)),
        # Fact(msl_pressure=P(lambda x: 1005 <= x <= 1010)),
        # Fact(uv_idx=P(lambda x: x < 3))
    )
    def recommend_cycling(self):
        self.declare(Fact(activity="*Cycling"))

    @Rule(
        Fact(t_max=P(lambda x: 25 <= x < 30)),
        # Fact(t_min=P(lambda x: x > 25)),
        # Fact(wind_gust=P(lambda x: x == 3)),
        # Fact(precip=P(lambda x: x == 0)),
        # Fact(msl_pressure=P(lambda x: 1020 <= x <= 1025)),
        # Fact(uv_idx=P(lambda x: x == 5))
    )
    def recommend_picnicking(self):
        self.declare(Fact(activity="*Picnicking"))

    @Rule(
        Fact(t_max=P(lambda x: 10 <= x < 15)),
        # Fact(t_min=P(lambda x: x > 10)),
        # Fact(wind_gust=P(lambda x: x == 7)),
        # Fact(precip=P(lambda x: x == 1)),
        # Fact(msl_pressure=P(lambda x: 1010 <= x <= 1015)),
        # Fact(uv_idx=P(lambda x: 3 <= x <= 4))
    )
    def recommend_hiking(self):
        self.declare(Fact(activity="*Hiking"))

    @Rule(
        Fact(t_max=P(lambda x: 0 <= x < 5)),
        # Fact(t_min=P(lambda x: x > 0)),
        # Fact(wind_gust=P(lambda x: x == 8)),
        # Fact(precip=P(lambda x: x == 2)),
        # Fact(msl_pressure=P(lambda x: 1015 <= x <= 1020)),
        # Fact(uv_idx=P(lambda x: 5 <= x <= 6))
    )
    def recommend_bird_watching(self):
        self.declare(Fact(activity="*Bird Watching"))

    @Rule(
        Fact(t_max=P(lambda x: 5 <= x < 10)),
        # Fact(t_min=P(lambda x: x > 5)),
        # Fact(wind_gust=P(lambda x: x < 5)),
        # Fact(precip=P(lambda x: x == 4)),
        # Fact(msl_pressure=P(lambda x: 1025 <= x <= 1030)),
        # Fact(uv_idx=P(lambda x: 4 <= x <= 5))
    )
    def recommend_fishing(self):
        self.declare(Fact(activity="*Fishing"))

    @Rule(
        Fact(t_max=P(lambda x: 30 <= x < 32)),
        # Fact(t_min=P(lambda x: x > 30)),
        # Fact(wind_gust=P(lambda x: x == 5)),
        # Fact(precip=P(lambda x: x == 0)),
        # Fact(msl_pressure=P(lambda x: 1000 <= x <= 1005)),
        # Fact(uv_idx=P(lambda x: x < 2))
    )
    def recommend_markets(self):
        self.declare(Fact(activity="*Beach"))

    @Rule(
        Fact(t_max=P(lambda x: 20 <= x < 25)),
        # Fact(t_min=P(lambda x: x > 20)),
        # Fact(wind_gust=P(lambda x: x == 2)),
        # Fact(precip=P(lambda x: x == 0)),
        # Fact(msl_pressure=P(lambda x: 1030 <= x <= 1032)),
        Fact(uv_idx=P(lambda x: 0 <= x <= 1))
    )
    def recommend_stargazing(self):
        self.declare(Fact(activity="*Stargazing"))

