"""Domain vocabulary used to synthesise reviews and to drive dictionary NER.

Kept in one place so the dataset generator and the custom NER cannot drift apart.
"""

from __future__ import annotations

POSITIVE_WORDS: tuple[str, ...] = (
    'excellent', 'amazing', 'great', 'good', 'fantastic', 'wonderful', 'brilliant',
    'perfect', 'outstanding', 'superb', 'masterpiece', 'stunning', 'impressive',
    'enjoyable', 'entertaining', 'captivating', 'engaging', 'powerful', 'moving',
    'beautiful', 'compelling', 'memorable', 'remarkable', 'spectacular', 'phenomenal',
)

NEGATIVE_WORDS: tuple[str, ...] = (
    'terrible', 'awful', 'bad', 'poor', 'disappointing', 'boring', 'dull',
    'mediocre', 'waste', 'horrible', 'worst', 'stupid', 'annoying', 'predictable',
    'unbearable', 'ridiculous', 'failure', 'disaster', 'nonsense', 'mess',
    'underwhelming', 'forgettable', 'confusing', 'pointless', 'painful',
)

DIRECTOR_NAMES: tuple[str, ...] = (
    'Steven Spielberg', 'Christopher Nolan', 'Martin Scorsese', 'Quentin Tarantino',
    'James Cameron', 'Kathryn Bigelow', 'Alfred Hitchcock', 'Ridley Scott',
    'Greta Gerwig', 'Sofia Coppola', 'Denis Villeneuve', 'Francis Ford Coppola',
    'David Fincher', 'Spike Lee', 'Wes Anderson', 'Ava DuVernay',
)

ACTOR_NAMES: tuple[str, ...] = (
    'Tom Hanks', 'Meryl Streep', 'Leonardo DiCaprio', 'Jennifer Lawrence',
    'Denzel Washington', 'Viola Davis', 'Brad Pitt', 'Cate Blanchett',
    'Robert De Niro', 'Kate Winslet', 'Morgan Freeman', 'Scarlett Johansson',
    'Daniel Day-Lewis', 'Emma Stone', 'Samuel L. Jackson', 'Natalie Portman',
)

MOVIE_TITLES: tuple[str, ...] = (
    'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'The Dark Knight',
    "Schindler's List", 'Forrest Gump', 'Inception', 'The Matrix',
    'Titanic', 'Avatar', 'Parasite', 'Casablanca',
    'Goodfellas', 'The Silence of the Lambs', 'Jurassic Park', 'Star Wars',
)

AWARD_NAMES: tuple[str, ...] = (
    'Oscar', 'Academy Award', 'Golden Globe', 'BAFTA',
    "Palme d'Or", 'Emmy', 'Screen Actors Guild Award', 'Tony Award',
    "Critics' Choice", 'Independent Spirit Award', 'Cesar Award', 'Goya Award',
)

MOVIE_TERMS: tuple[str, ...] = (
    'movie', 'film', 'cinema', 'director', 'actor', 'actress',
    'script', 'screenplay', 'scene', 'plot', 'character', 'performance',
)

#: Maps an entity label to the gazetteer backing it. Order matters: when two
#: gazetteer hits cover the same span, the earlier label wins (see ner.py).
GAZETTEERS: dict[str, tuple[str, ...]] = {
    'DIRECTOR': DIRECTOR_NAMES,
    'ACTOR': ACTOR_NAMES,
    'MOVIE': MOVIE_TITLES,
    'AWARD': AWARD_NAMES,
}
