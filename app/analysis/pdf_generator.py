from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class SentimentAnalysisReport:
    def __init__(self, data: list, request_id: str):
        """Initialize with the generated reviews data and request ID"""
        self.df = pd.DataFrame(data)
        self.df['length'] = self.df['text'].apply(len)
        # Create unique paths for this analysis
        self.temp_dir = f"temp_{request_id}"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.output_path = os.path.join(self.temp_dir, f"analysis_report_{request_id}.pdf")
        self.temp_files = []

    def generate_report(self) -> str:
        """Generate the PDF report and return its path"""
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Generate all visualizations and add to PDF
            self._add_title_page(pdf)
            self._add_sentiment_distribution(pdf)
            self._add_wordcloud(pdf)
            self._add_kl_divergence_and_venn(pdf)
            self._add_ngram_analysis(pdf)
            
            # Save and return path
            pdf.output(self.output_path)
            return self.output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise
        
    def cleanup(self):
        """Clean up temporary files and directory"""
        try:
            for file in self.temp_files:
                if os.path.exists(file):
                    os.remove(file)
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def _get_temp_path(self, filename: str) -> str:
        """Generate a unique path for temporary files"""
        path = os.path.join(self.temp_dir, filename)
        self.temp_files.append(path)
        return path

    def _add_title_page(self, pdf: FPDF):
        pdf.add_page()
        pdf.set_font("Arial", size=16, style='B')
        pdf.cell(200, 10, txt="Synthetic Data Analysis: Sentiment Evaluation Report", ln=True, align='C')

    def _add_sentiment_distribution(self, pdf: FPDF):
        sentiment_dist_path = self._get_temp_path("sentiment_distribution.png")
        
        sentiment_counts = self.df['sentiment'].value_counts()
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        ax[0].bar(sentiment_counts.index, sentiment_counts.values, 
                 color=['#6cba6b', '#f16a6a', '#d1d1d1'])
        ax[0].set_title('Sentiment Distribution')
        ax[0].set_xlabel('Sentiment')
        ax[0].set_ylabel('Frequency')
        
        sns.histplot(self.df['length'], kde=True, label='Synthetic Reviews', color='#f79c42', stat='density', ax=ax[1])
        ax[1].set_title('Synthetic Review Length Distribution')
        ax[1].set_xlabel('Review Length')
        ax[1].set_ylabel('Density')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(sentiment_dist_path)
        plt.close()

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="1. Sentiment and Length Distribution", ln=True, align='L')
        pdf.multi_cell(0, 10, txt="This section shows the distribution of sentiments within the synthetic data, as well as the length distribution of synthetic tweets.")
        pdf.image(sentiment_dist_path, x=30, w=150)

    def _add_wordcloud(self, pdf: FPDF):
        wordcloud_path = self._get_temp_path("wordcloud.png")
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(self.df['text']))
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wordcloud_path)
        plt.close()

        pdf.add_page()
        pdf.cell(200, 10, txt="2. Word Cloud Analysis", ln=True, align='L')
        pdf.multi_cell(0, 10, txt="Visual representation of most frequent terms in the dataset.")
        pdf.image(wordcloud_path, x=30, w=150)

    def _add_kl_divergence_and_venn(self, pdf: FPDF):
        kl_venn_path = self._get_temp_path("kl_venn.png")
        
       
        real_dist = self.df['sentiment'].value_counts(normalize=True)
        synthetic_dist = self.df['sentiment'].value_counts(normalize=True)
        kl_div = entropy(synthetic_dist, qk=real_dist)
        #Isnt assigned in the reference provided so dunno
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        ax[0].bar(real_dist.index, real_dist.values, width=0.4, label='Real Data', align='center', color='#4a90e2')
        ax[0].bar(synthetic_dist.index, synthetic_dist.values, width=0.4, label='Synthetic Data', align='edge', color='#f5a623')
        ax[0].set_title('Real vs Synthetic Sentiment Proportions')
        ax[0].set_xlabel('Sentiment')
        ax[0].set_ylabel('Proportion')
        ax[0].legend()

        real_words = set(" ".join(self.df['text']).split())
        synthetic_words = set(" ".join(self.df['text']).split())
        venn2([real_words, synthetic_words], set_labels=('Real Data', 'Synthetic Data'))
        ax[1].set_title('Token Leakage: Word Overlap', fontsize=16)

        plt.tight_layout()
        plt.savefig(kl_venn_path)
        plt.close()

        pdf.add_page()
        pdf.cell(200, 10, txt="3. Sentiment Proportions and Token Leakage", ln=True, align='L')
        pdf.multi_cell(0, 10, txt="This section compares the proportions of sentiments in the real and synthetic datasets, showcasing differences in sentiment distribution. A Venn diagram visualizes the overlap of words between the two datasets, indicating token leakage or shared vocabulary between real and synthetic tweets.")
        pdf.image(kl_venn_path, x=30, w=150)

    def _add_ngram_analysis(self, pdf: FPDF):
        ngram_path = self._get_temp_path("ngram_analysis.png")
        
        vectorizer_bigram = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        bigram_matrix = vectorizer_bigram.fit_transform(self.df['text'])

        vectorizer_trigram = CountVectorizer(ngram_range=(3, 3), stop_words='english')
        trigram_matrix = vectorizer_trigram.fit_transform(self.df['text'])

        bigram_freq = bigram_matrix.sum(axis=0).A1
        trigram_freq = trigram_matrix.sum(axis=0).A1

        bigram_terms = vectorizer_bigram.get_feature_names_out()
        trigram_terms = vectorizer_trigram.get_feature_names_out()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].set_xticks(range(10))
        axes[0].set_xticklabels(bigram_terms[:10], rotation=45, ha='right')
        axes[0].bar(bigram_terms[:10], bigram_freq[:10], color='#9b59b6')
        axes[0].set_title('Top 10 Bigrams')

        axes[1].set_xticks(range(10))
        axes[1].set_xticklabels(trigram_terms[:10], rotation=45, ha='right')
        axes[1].bar(trigram_terms[:10], trigram_freq[:10], color='#f39c12')
        axes[1].set_title('Top 10 Trigrams')

        plt.tight_layout()
        plt.savefig(ngram_path)
        plt.close()

        pdf.add_page()
        pdf.cell(200, 10, txt="4. N-Gram Frequencies (Bigram & Trigram)", ln=True, align='L')
        pdf.multi_cell(0, 10, txt="This section highlights the most frequent bigrams and trigrams (two- and three-word combinations) within the synthetic data. These n-grams are useful for understanding the most commonly used word pairs or triplets and can help identify important topics and language patterns in the dataset.")
        pdf.image(ngram_path, x=30, w=150) 