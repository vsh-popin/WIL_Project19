from experta import *

class WeatherRecommender(KnowledgeEngine):
    
    @Rule(
        Fact(t_max=P(lambda x: x > 25)),
        # Fact(wind_gust=P(lambda x: x < 15)),
        Fact(uv_idx=P(lambda x: x >= 5 and x <= 8))
    )
    def recommend_beach(self):
        self.declare(Fact(activity="Beach"))

    @Rule(
        Fact(t_max=P(lambda x: x > 15 and x <= 25)),
        # Fact(wind_gust=P(lambda x: x < 20)),
        Fact(uv_idx=P(lambda x: x >= 3 and x <= 6))
    )
    def recommend_hiking(self):
        self.declare(Fact(activity="Hiking"))

    @Rule(
        # Fact(wind_gust=P(lambda x: x > 25)),
        Fact(uv_idx=P(lambda x: x > 8))
    )
    def recommend_indoor(self):
        self.declare(Fact(activity="Indoor Activities"))

    @Rule(
        Fact(t_max=P(lambda x: x <= 5)),
        # Fact(wind_gust=P(lambda x: x < 20))
    )
    def recommend_skiing(self):
        self.declare(Fact(activity="Skiing"))

    @Rule(
        Fact(t_max=P(lambda x: x > 20 and x <= 30)),
        # Fact(wind_gust=P(lambda x: x < 15)),
        Fact(uv_idx=P(lambda x: x >= 3 and x <= 6))
    )
    def recommend_cycling(self):
        self.declare(Fact(activity="Cycling"))

    @Rule(
        Fact(t_max=P(lambda x: x > 10 and x <= 20)),
        # Fact(wind_gust=P(lambda x: x < 10)),
        Fact(uv_idx=P(lambda x: x >= 2 and x <= 5))
    )
    def recommend_running(self):
        self.declare(Fact(activity="Running"))

    @Rule(
        Fact(uv_idx=P(lambda x: x <= 0.0))  # UV index is 0
    )
    def recommend_stargazing(self):
        self.declare(Fact(activity="Stargazing"))

    # @Rule(
    #     Fact(uv_idx=P(lambda x: x > 1 and x < 4))
    # )
    # def recommend_stargazing(self):
    #     self.declare(Fact(activity="Shopping"))