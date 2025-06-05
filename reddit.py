import praw
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time


class RedditSentimentAnalyzer:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent)
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def get_posts(self, subreddit_name, limit=100, sort_type='hot'):
        print(f"Fetching {limit} {sort_type} posts from r/{subreddit_name}")

        subreddit = self.reddit.subreddit(subreddit_name)
        post_data = []

                                                     # Get filtered posts
        if sort_type == 'hot':
            posts = subreddit.hot(limit=limit)
        elif sort_type == 'top':
            time_filter = input("Choose time filter (day, week, month, year, all): ").strip().lower()
            posts = subreddit.top(limit=limit, time_filter=time_filter)
        elif sort_type == 'new':
            posts = subreddit.new(limit=limit)
        else:
            posts = subreddit.hot(limit=limit)

        for post in posts:
            if post.stickied:                            # Skip pinned posts
                continue

            post_data.append({
                'title': post.title,
                'subreddit': subreddit_name,
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'author': str(post.author) if post.author else '[deleted]',
                'selftext': post.selftext[:500] if post.selftext else '',
                'url': post.url,
            })

        return pd.DataFrame(post_data)

    def analyze_sentiments(self, text):

        if not text or pd.isna(text):
            return 0, 0, 0, 0

                                                                     # Get VADER scores
        scores = self.vader_analyzer.polarity_scores(str(text))

                                                                     # Return compound, positive, negative, neutral
        return (scores['compound'],
                scores['pos'],
                scores['neg'],
                scores['neu'])

    def process_df(self, df):
        print("Analyzing the posts with VADER...")

        df['full_text'] = df['title'] + " " + df['selftext']

                                                                             # Apply VADER sentiment analysis
        sentiment_results = df['full_text'].apply(self.analyze_sentiments)

                                                                             # Extract VADER scores
        df['compound'] = [result[0] for result in sentiment_results]
        df['positive'] = [result[1] for result in sentiment_results]
        df['negative'] = [result[2] for result in sentiment_results]
        df['neutral'] = [result[3] for result in sentiment_results]

                                                                     # Create sentiment labels based on compound score
        df['sentiment_label'] = df['compound'].apply(self.categorize_sents)

                                                                    # Add time features
        df['hour'] = df['created_utc'].dt.hour
        df['day'] = df['created_utc'].dt.day_name()

        return df

    def categorize_sents(self, compound_score):

        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def create_visualizations(self, df, subreddit_name):

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Reddit Sentiment Analysis for r/{subreddit_name}', fontsize=16)

                                                                                                # 1. Sentiment distribution (pie chart)
        sentiment_counts = df['sentiment_label'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                       autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightblue'])
        axes[0, 0].set_title('Overall Sentiment Distribution')

                                                                                                 # 2. Compound score vs Post Score scatter plot
        axes[0, 1].scatter(df['compound'], df['score'], alpha=0.6, c=df['compound'],
                           cmap='RdYlGn', s=50)
        axes[0, 1].set_xlabel('VADER Compound Score')
        axes[0, 1].set_ylabel('Post Score (Upvotes)')
        axes[0, 1].set_title('Sentiment vs Post Popularity')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)

                                                                                                 # 3. VADER scores distribution (histogram)
        axes[0, 2].hist(df['compound'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].set_xlabel('VADER Compound Score')
        axes[0, 2].set_ylabel('Number of Posts')
        axes[0, 2].set_title('Distribution of Compound Scores')
        axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.5)

                                                                                                  # 4. Sentiment by hour of day
        hourly_sentiment = df.groupby('hour')['compound'].mean()
        axes[1, 0].plot(hourly_sentiment.index, hourly_sentiment.values,
                        marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Compound Score')
        axes[1, 0].set_title('Sentiment Throughout the Day')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Positive vs Negative scores scatter
        axes[1, 1].scatter(df['positive'], df['negative'], alpha=0.6,
                           c=df['compound'], cmap='RdYlGn', s=50)
        axes[1, 1].set_xlabel('Positive Score')
        axes[1, 1].set_ylabel('Negative Score')
        axes[1, 1].set_title('Positive vs Negative Intensity')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Enhanced summary with VADER metrics
        top_positive = df.nlargest(3, 'compound')[['title', 'compound', 'score']]
        top_negative = df.nsmallest(3, 'compound')[['title', 'compound', 'score']]

        axes[1, 2].axis('off')
        summary_text = f"""
        ANALYSIS SUMMARY:

        Total Posts: {len(df)}

        Sentiment Breakdown:
        • Positive: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0) / len(df) * 100:.1f}%)
        • Neutral: {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0) / len(df) * 100:.1f}%)
        • Negative: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0) / len(df) * 100:.1f}%)

        VADER Scores:
        • Avg Compound: {df['compound'].mean():.3f}
        • Avg Positive: {df['positive'].mean():.3f}
        • Avg Negative: {df['negative'].mean():.3f}
        • Avg Neutral: {df['neutral'].mean():.3f}

        Most Positive:
        "{top_positive.iloc[0]['title'][:50]}..."
        (Score: {top_positive.iloc[0]['compound']:.3f})

        Most Negative:
        "{top_negative.iloc[0]['title'][:50]}..."
        (Score: {top_negative.iloc[0]['compound']:.3f})
        """

        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=9, fontfamily='monospace')

        plt.tight_layout()
        plt.show()
        return fig

    def get_insights(self, df):
        print("\n" + "_" * 60)
        print(' REDDIT SENTIMENT ANALYSIS INSIGHTS')
        print("_" * 60)

        # Basic stats
        total_posts = len(df)
        avg_compound = df['compound'].mean()
        avg_positive = df['positive'].mean()
        avg_negative = df['negative'].mean()
        avg_neutral = df['neutral'].mean()

        print(f"📊 Analyzed {total_posts} posts")
        print(f" Average compound score: {avg_compound:.3f}")
        print(f" Average positive intensity: {avg_positive:.3f}")
        print(f" Average negative intensity: {avg_negative:.3f}")
        print(f" Average neutral intensity: {avg_neutral:.3f}")

        # Sentiment distribution
        sentiment_dist = df['sentiment_label'].value_counts(normalize=True) * 100
        print(f"\n💥 Sentiment Distribution:")
        for sentiment, percentage in sentiment_dist.items():
            emoji = "😊" if sentiment == "Positive" else "😢" if sentiment == "Negative" else "😐"
            print(f"   {emoji} {sentiment}: {percentage:.1f}%")

        # VADER-specific insights
        very_positive = len(df[df['compound'] > 0.5])
        very_negative = len(df[df['compound'] < -0.5])
        print(f"\n🔥 Very positive posts (>0.5): {very_positive} ({very_positive / len(df) * 100:.1f}%)")
        print(f"💀 Very negative posts (<-0.5): {very_negative} ({very_negative / len(df) * 100:.1f}%)")

        # Correlation insights
        score_correlation = df['compound'].corr(df['score'])
        comment_correlation = df['compound'].corr(df['num_comments'])

        print(f"\n🔗 Correlation with upvotes: {score_correlation:.3f}")
        print(f"💬 Correlation with comments: {comment_correlation:.3f}")

        # Time-based insights
        best_hour = df.groupby('hour')['compound'].mean().idxmax()
        worst_hour = df.groupby('hour')['compound'].mean().idxmin()
        print(f"\n⏰ Most positive hour: {best_hour}:00")
        print(f"⏰ Most negative hour: {worst_hour}:00")

        # Top posts
        print(f"\n🏆 Most positive post:")
        top_pos = df.loc[df['compound'].idxmax()]
        print(f"   '{top_pos['title'][:80]}...'")
        print(f"   VADER Score: {top_pos['compound']:.3f}, Upvotes: {top_pos['score']}")

        print(f"\n💔 Most negative post:")
        top_neg = df.loc[df['compound'].idxmin()]
        print(f"   '{top_neg['title'][:80]}...'")
        print(f"   VADER Score: {top_neg['compound']:.3f}, Upvotes: {top_neg['score']}")

        print(f"\n📚 Score Interpretation:")
        print(f"   • Compound > 0.05: Positive sentiment")
        print(f"   • Compound < -0.05: Negative sentiment")
        print(f"   • -0.05 ≤ Compound ≤ 0.05: Neutral sentiment")
        print(f"   • Scores closer to ±1 indicate stronger sentiment")


def testing():
    print("\n" + "_" * 60)
    print("🚀 Reddit Sentiment Analyzer by grimroze 👌")
    print("_" * 60)

    CLIENT_ID = "8904c@OR-2-2cr-c@IR)C@"
    CLIENT_SECRET = "894urvc39ujc3092r82802@#$"                                 # its sample, add you own credentials here
    USER_AGENT = "vader_sentiment_analyzer_v2.0_by_grimroze"

    subreddit_name = input("Enter the subreddit name (without r/): ").strip()

    while True:
        try:
            limit = int(input("Enter the number of posts to fetch (max 100): "))
            if 1 <= limit <= 100:
                break
            else:
                print("\n❌ Limit must be between 1 and 100")
        except ValueError:
            print("❌ Invalid value, please try again")

    sort_choice = input("Enter the sort type (hot/new/top): ").strip().lower()
    if sort_choice not in ['hot', 'new', 'top']:
        sort_choice = 'hot'
        print(f"  Invalid sort type, defaulting to 'hot'")

    print("\n Initializing  analyzer...")
    analyzer = RedditSentimentAnalyzer(CLIENT_ID, CLIENT_SECRET, USER_AGENT)

    print(" Fetching posts...")
    df = analyzer.get_posts(subreddit_name, limit=limit, sort_type=sort_choice)

    print(" Processing sentiment analysis...")
    df = analyzer.process_df(df)

    print(" Creating visualizations...")
    analyzer.create_visualizations(df, subreddit_name)

    analyzer.get_insights(df)

    output_file = f'{subreddit_name}_vader_sentiment_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to {output_file}")
    print("✅ Analysis complete!")


if __name__ == "__main__":
    testing()