"""Task 2 — data filtering and selection."""

from __future__ import annotations

from pandas_homework import filtering


class TestReleasedAfter:

    def test_is_strictly_after_the_given_year(self, tiny_movies):
        titles = set(filtering.released_after(tiny_movies, 2003)['Title'])
        assert titles == {'C', 'D', 'E'}

    def test_excludes_the_boundary_year(self, tiny_movies):
        assert 'B' not in set(filtering.released_after(tiny_movies, 2003)['Title'])

    def test_matches_the_brief_on_generated_data(self, raw_movies):
        result = filtering.released_after(raw_movies, 2010)
        assert (result['Year'] > 2010).all()
        assert len(result) == (raw_movies['Year'] > 2010).sum()


class TestRatedAbove:

    def test_is_strictly_above_the_threshold(self, tiny_movies):
        titles = set(filtering.rated_above(tiny_movies, 7.0)['Title'])
        assert titles == {'C', 'F'}

    def test_excludes_rows_with_a_missing_rating(self, tiny_movies):
        """Row D has no rating; NaN > x is False, so it must not appear."""
        assert 'D' not in set(filtering.rated_above(tiny_movies, 0.0)['Title'])

    def test_returns_nothing_above_the_maximum(self, raw_movies):
        assert filtering.rated_above(raw_movies, 10.0).empty


class TestInGenres:

    def test_selects_every_listed_genre(self, tiny_movies):
        result = filtering.in_genres(tiny_movies, ['Action', 'Comedy'])
        assert set(result['Title']) == {'A', 'B', 'C', 'D'}

    def test_accepts_a_single_genre(self, tiny_movies):
        assert set(filtering.in_genres(tiny_movies, ['Drama'])['Title']) == {'E', 'F'}

    def test_unknown_genre_yields_nothing(self, tiny_movies):
        assert filtering.in_genres(tiny_movies, ['Documentary']).empty

    def test_empty_genre_list_yields_nothing(self, tiny_movies):
        assert filtering.in_genres(tiny_movies, []).empty


class TestBoxOfficeExceedsBudgetMultiple:

    def test_finds_rows_beating_the_multiple(self, tiny_movies):
        # A: 300 > 2*100 -> True. C: 500 > 2*200 -> True. B: 40 > 2*50 -> False.
        result = filtering.box_office_exceeds_budget_multiple(tiny_movies, 2.0)
        assert set(result['Title']) == {'A', 'C'}

    def test_excludes_the_exact_boundary(self, tiny_movies):
        """F earned exactly 2x its budget (160 on 80), and the brief says
        'greater than', so it must fall outside the result."""
        assert 'F' not in set(
            filtering.box_office_exceeds_budget_multiple(tiny_movies, 2.0)['Title'])

    def test_excludes_rows_missing_either_figure(self, tiny_movies):
        """D has box office but no budget; the comparison is undefined."""
        result = filtering.box_office_exceeds_budget_multiple(tiny_movies, 0.1)
        assert 'D' not in set(result['Title'])

    def test_multiple_is_configurable(self, tiny_movies):
        result = filtering.box_office_exceeds_budget_multiple(tiny_movies, 1.0)
        assert set(result['Title']) == {'A', 'C', 'F'}


class TestByDirectors:

    def test_matches_full_names_exactly(self, tiny_movies):
        result = filtering.by_directors(tiny_movies)
        assert set(result['Title']) == {'A', 'C'}

    def test_does_not_match_a_shared_first_name(self, tiny_movies):
        """Christopher Brown and Steven Taylor are different people."""
        result = filtering.by_directors(tiny_movies)
        assert {'B', 'D'}.isdisjoint(set(result['Title']))

    def test_returns_nothing_on_generated_data(self, raw_movies):
        """REGRESSION: the brief asks for Nolan and Spielberg, but its own
        generator pairs random first and last names and can never produce
        either. The classroom notebook silently answered a different question
        by substituting a first-name substring match."""
        assert filtering.by_directors(raw_movies).empty

    def test_first_name_mode_is_opt_in_and_broader(self, tiny_movies):
        result = filtering.by_directors(tiny_movies, match_first_name_only=True)
        assert set(result['Title']) == {'A', 'B', 'C', 'D'}

    def test_first_name_mode_matches_many_directors_on_generated_data(self, raw_movies):
        result = filtering.by_directors(raw_movies, match_first_name_only=True)
        assert result['Director'].nunique() > 2

    def test_first_name_mode_does_not_match_mid_token(self, tiny_movies):
        """'Steven' as a substring would also hit a surname containing it."""
        frame = tiny_movies.assign(Director=['Stevenson Ray'] + ['Zed Zed'] * 5)
        result = filtering.by_directors(
            frame, ['Steven Spielberg'], match_first_name_only=True)
        assert result.empty

    def test_custom_director_list(self, tiny_movies):
        result = filtering.by_directors(tiny_movies, ['Greta Davis'])
        assert set(result['Title']) == {'E', 'F'}

    def test_empty_director_list_yields_nothing(self, tiny_movies):
        assert filtering.by_directors(tiny_movies, []).empty


class TestImmutability:

    def test_no_filter_mutates_its_input(self, tiny_movies):
        before = tiny_movies.copy()
        filtering.released_after(tiny_movies, 2000)
        filtering.rated_above(tiny_movies, 5.0)
        filtering.in_genres(tiny_movies, ['Action'])
        filtering.box_office_exceeds_budget_multiple(tiny_movies)
        filtering.by_directors(tiny_movies)
        assert tiny_movies.equals(before)
