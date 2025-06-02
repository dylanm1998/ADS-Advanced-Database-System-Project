from pymongo import MongoClient
import pandas as pd
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource

output_notebook()


def populate_data_if_needed(users_col, movies_col, ratings_col):
    """
    Load raw MovieLens data into MongoDB if collections are empty.
    """
    # If any of the three collections already has documents, skip loading entirely.
    if (users_col.estimated_document_count() > 0 and movies_col.estimated_document_count() > 0 and ratings_col.estimated_document_count() > 0):
        print("Data already exists in MongoDB collections. Skipping data load.")
        return

    # 1) Load users
    users_df = pd.read_csv("ml-100k/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"],)
    try:
        users_col.insert_many(users_df.to_dict("records"))
    except Exception as e:
        print(f"Warning: could not insert into users collection: {e}")

    # 2) Load movies (first 5 columns + 19 binary genre flags)
    movies_cols = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + [f"genre_{i}" for i in range(19)]
    movies_df = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", usecols=range(24), names=movies_cols)
    try:
        movies_col.insert_many(movies_df.to_dict("records"))
    except Exception as e:
        print(f"Warning: could not insert into movies collection: {e}")

    # 3) Load ratings
    ratings_df = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"],)
    try:
        ratings_col.insert_many(ratings_df.to_dict("records"))
    except Exception as e:
        print(f"Warning: could not insert into ratings collection: {e}")

    print("Data population complete.")


def populate_user_movie_info_if_needed(userinfo_genres_col, ratings_col):
    """
    Create a new collection that combines ratings with user info and movie genres.
    """
    # If already populated, skip
    if userinfo_genres_col.estimated_document_count() > 0:
        print("ratings_userinfo_genres already has data. Skipping join step.")
        return

    pipeline = [
        {
            "$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "user_id",
                "as": "user_info",
            }
        },
        {"$unwind": "$user_info"},
        {
            "$lookup": {
                "from": "movies",
                "localField": "movie_id",
                "foreignField": "movie_id",
                "as": "movie_info",
            }
        },
        {"$unwind": "$movie_info"},
        {
            "$project": {
                "_id": 0,
                "user_id": 1,
                "movie_id": 1,
                "rating": 1,
                "age": "$user_info.age",
                "gender": "$user_info.gender",
                "occupation": "$user_info.occupation",
                # Copy all 19 genre flags from movie_info
                **{f"genre_{i}": f"$movie_info.genre_{i}" for i in range(19)},
            }
        },
        {"$out": "ratings_userinfo_genres"},  # directly writes into new collection
    ]

    # Run the aggregation that writes to ratings_userinfo_genres
    try:
        ratings_col.aggregate(pipeline, allowDiskUse=True)
    except Exception as e:
        print(f"Error during aggregation into ratings_userinfo_genres: {e}")
        return

    print("ratings_userinfo_genres has been created and populated.")


def build_and_insert_stats(source_col, target_col, group_field: str, genre_count: int = 19):
    """
    Use MongoDB aggregation to compute average rating and count per (group_field, genre_index).
    Inserts results directly into target_col.
    """
    # If target already has data, skip
    if target_col.estimated_document_count() > 0:
        print(f"{target_col.name} already has data. Skipping stats aggregation.")
        return

    # We unwind each genre flag and filter out only those where genre_flag == 1
    # Then group by (group_field, genre_index) to compute avg & count.
    pipeline = [
        {
            # Project group_field and an array of {index, flag} for each genre
            "$project": {
                group_field: 1,
                "rating": 1,
                "genre_flags": [
                    {"index": i, "flag": f"$genre_{i}"} for i in range(genre_count)
                ],
            }
        },
        {"$unwind": "$genre_flags"},
        {"$match": {"genre_flags.flag": 1}},
        {
            "$group": {
                "_id": {group_field: f"${group_field}", "genre_index": "$genre_flags.index"},
                "avg_rating": {"$avg": "$rating"},
                "count": {"$sum": 1},
            }
        },
        {
            # reshape the document for insertion
            "$project": {
                "_id": 0,
                group_field: "$_id." + group_field,
                "genre_index": "$_id.genre_index",
                "avg_rating": {"$round": ["$avg_rating", 3]},
                "count": 1,
            }
        },
        {"$out": target_col.name},
    ]

    try:
        source_col.aggregate(pipeline, allowDiskUse=True)
        print(f"Statistics for {group_field} inserted into {target_col.name}.")
    except Exception as e:
        print(f"Error creating stats for {group_field}: {e}")


def populate_statistics_if_needed(userinfo_genres_col, age_stats_col, gender_stats_col, occupation_stats_col):
    """
    Populate age, gender, and occupation statistics collections by calling build_and_insert_stats.
    """
    build_and_insert_stats(userinfo_genres_col, age_stats_col, group_field="age")
    build_and_insert_stats(userinfo_genres_col, gender_stats_col, group_field="gender")
    build_and_insert_stats(userinfo_genres_col, occupation_stats_col, group_field="occupation")


def age_to_group(age: int) -> str:
    """
    Convert a numeric age into a decade‐range string, e.g. 23 -> "20-29".
    """
    if age < 10:
        return "0-9"
    lower = (age // 10) * 10
    return f"{lower}-{lower + 9}"


def make_bokeh_charts(doc_df: pd.DataFrame, group_field: str, group_values: list, genre_labels: list, output_html: str):
    """
    Given a DataFrame with columns [group_field, genre_index, avg_rating, count],
    produce a grid of Bokeh charts: two vertical bars per group_value.
    """
    plots = []

    for val in group_values:
        subset = doc_df[doc_df[group_field] == val]
        if subset.empty:
            continue

        subset = subset.copy()
        subset["genre"] = subset["genre_index"].apply(lambda idx: genre_labels[idx])

        # Plot 1: average rating per genre
        p1 = figure(x_range=genre_labels, height=300, width=600, title=f"{group_field.capitalize()}: {val} — Avg Rating", toolbar_location=None,tools="",)
        src1 = ColumnDataSource(subset)
        p1.vbar(x="genre", top="avg_rating", width=0.8, source=src1)
        p1.xaxis.major_label_orientation = 1.2
        p1.yaxis.axis_label = "Avg Rating"
        p1.y_range.start = 0

        # Plot 2: count per genre
        p2 = figure(x_range=genre_labels, height=300, width=600, title=f"{group_field.capitalize()}: {val} — Count", toolbar_location=None, tools="",)
        src2 = ColumnDataSource(subset)
        p2.vbar(x="genre", top="count", width=0.8, source=src2)
        p2.xaxis.major_label_orientation = 1.2
        p2.yaxis.axis_label = "Count"
        p2.y_range.start = 0

        plots.append([p1, p2])

    if plots:
        output_file(output_html)
        show(gridplot(plots))


def plot_statistics(age_stats_col, gender_stats_col, occupation_stats_col, genre_labels):
    """
    Fetch each statistics collection into a DataFrame, then call make_bokeh_charts.
    """
    # 1) Plot Age Groups
    age_docs = list(age_stats_col.find({}, {"_id": 0}))
    if age_docs:
        df_age = pd.DataFrame(age_docs)
        df_age["age_group"] = df_age["age"].apply(age_to_group)
        all_age_groups = sorted(df_age["age_group"].unique())
        # Remap DataFrame so that grouping is by "age_group"
        df_age = df_age[["age_group", "genre_index", "avg_rating", "count"]].rename(columns={"age_group": "group"})
        make_bokeh_charts(df_age, group_field="group", group_values=all_age_groups, genre_labels=genre_labels, output_html="age_group_charts.html")

    # 2) Plot Gender
    gender_docs = list(gender_stats_col.find({}, {"_id": 0}))
    if gender_docs:
        df_gender = pd.DataFrame(gender_docs)
        df_gender["gender"] = df_gender["gender"].astype(str)
        all_genders = sorted(df_gender["gender"].unique())
        df_gender = df_gender[["gender", "genre_index", "avg_rating", "count"]].rename(columns={"gender": "group"})
        make_bokeh_charts(df_gender, group_field="group", group_values=all_genders, genre_labels=genre_labels, output_html="gender_charts.html")

    # 3) Plot Occupation
    occ_docs = list(occupation_stats_col.find({}, {"_id": 0}))
    if occ_docs:
        df_occ = pd.DataFrame(occ_docs)
        all_occs = sorted(df_occ["occupation"].unique())
        df_occ = df_occ[["occupation", "genre_index", "avg_rating", "count"]].rename(columns={"occupation": "group"})
        make_bokeh_charts(df_occ, group_field="group", group_values=all_occs, genre_labels=genre_labels, output_html="occupation_charts.html",)


if __name__ == "__main__":
    # 1) Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["movielens_100k"]

    # 2) Reference raw collections
    users_col = db["users"]
    movies_col = db["movies"]
    ratings_col = db["ratings"]

    # 3) Collections for joined user+movie+rating info & statistics
    userinfo_genres_col = db["ratings_userinfo_genres"]
    age_stats_col = db["age_genre_rating_stats"]
    gender_stats_col = db["gender_genre_rating_stats"]
    occupation_stats_col = db["occupation_genre_rating_stats"]

    # 4) Define genre labels
    genre_labels = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    # 5) Data population
    populate_data_if_needed(users_col, movies_col, ratings_col)

    # 6) Join ratings with user info and movie genres
    populate_user_movie_info_if_needed(userinfo_genres_col, ratings_col)

    # 7) Compute and populate statistics in MongoDB directly
    populate_statistics_if_needed(userinfo_genres_col, age_stats_col, gender_stats_col, occupation_stats_col)

    # 8) Render Bokeh charts
    plot_statistics(age_stats_col, gender_stats_col, occupation_stats_col, genre_labels)

    print("All tasks completed successfully.")
