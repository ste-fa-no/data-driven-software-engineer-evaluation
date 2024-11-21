import os
import unittest
import warnings
from unittest.mock import MagicMock, patch
import spacy
from reportlab.pdfgen import canvas
import docproc as dp
from PyPDF2 import PdfReader


def create_valid_pdf(path):
    c = canvas.Canvas(path)
    c.drawString(100, 750, 'This is a test PDF.')
    c.save()


class TestPDFProcessing(unittest.TestCase):


    def setUp(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.test_dir = './test_pdfs'
        self.test_pdf_path = os.path.join(self.test_dir, 'test.pdf')
        self.nlp = spacy.load('en_core_web_trf')
        os.makedirs(self.test_dir, exist_ok=True)
        create_valid_pdf(self.test_pdf_path)
        self.mock_reader = MagicMock(spec=PdfReader)


    def tearDown(self):
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)


    def test_get_pdf_files(self):
        other_file = os.path.join(self.test_dir, 'test.txt')
        with open(other_file, 'w') as f:
            f.write('Not a PDF')
        pdf_files = dp.get_pdf_files(self.test_dir)
        self.assertIn('test.pdf', pdf_files)
        self.assertNotIn('test.txt', pdf_files)


    def test_load_file(self):
        reader = dp.load_file(self.test_pdf_path)
        self.assertIsInstance(reader, PdfReader)


    def test_load_file_invalid_extension(self):
        non_pdf_path = os.path.join(self.test_dir, 'test.txt')
        with open(non_pdf_path, 'w') as f:
            f.write('Not a PDF')
        reader = dp.load_file(non_pdf_path)
        self.assertIsNone(reader)


    @patch("docproc.PdfReader")
    def test_load_folder(self, MockPdfReader):
        mock_reader = MagicMock()
        MockPdfReader.return_value = mock_reader
        files = dp.load_folder(self.test_dir)
        MockPdfReader.assert_called_with(os.path.join(self.test_dir, 'test.pdf'))
        self.assertIn('test.pdf', files)
        self.assertEqual(files['test.pdf'], mock_reader)


    @patch("PyPDF2.PdfReader")
    def test_get_document_text(self, MockPdfReader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = 'Sample text.'
        MockPdfReader.pages = [mock_page]
        text = dp.get_document_text(MockPdfReader)
        self.assertEqual(text, 'Sample text.')


    def test_extract_entities(self):
        entities = dp.extract_entities(self.nlp, 'John Doe was born on January 1, 2000.')
        self.assertTrue(len(entities) >= 1)


    def test_get_sentences(self):
        sentences = list(dp.get_sentences(self.nlp, 'This is sentence one. This is sentence two.'))
        self.assertEqual(len(sentences), 2)


    def test_extract_emails(self):
        emails = dp.extract_emails(self.nlp, 'Contact us at test@example.com or support@domain.org.')
        self.assertEqual(emails, ['test@example.com', 'support@domain.org'])


    @patch("docproc.load_file")
    @patch("docproc.get_document_text")
    @patch("docproc.get_sentences")
    @patch("docproc.extract_emails")
    @patch("docproc.extract_dates")
    def test_extract_features_from_pdf(self, mock_extract_dates, mock_extract_emails, mock_get_sentences, mock_get_document_text, mock_load_file):
        mock_load_file.return_value = self.mock_reader

        mock_get_document_text.return_value = (
            'This is a sentence with an email test@example.com. '
            'Another sentence with a date January 1, 2024.'
        )

        mock_get_sentences.return_value = [
            MagicMock(text='This is a sentence with an email test@example.com.'),
            MagicMock(text='Another sentence with a date January 1, 2024.'),
        ]

        mock_extract_emails.side_effect = lambda nlp, text: ['test@example.com'] if 'email' in text else []
        mock_extract_dates.side_effect = lambda nlp, text: ['January 1, 2024'] if 'date' in text else []

        features = dp.extract_features_from_pdf(self.nlp, 'mock_file.pdf')

        self.assertEqual(len(features), 2)

        self.assertEqual(features[0], {
            'dates': [],
            'email_addresses': ['test@example.com'],
            'text': 'This is a sentence with an email test@example.com.'
        })

        self.assertEqual(features[1], {
            'dates': ['January 1, 2024'],
            'email_addresses': [],
            'text': 'Another sentence with a date January 1, 2024.'
        })

        mock_load_file.assert_called_once_with('mock_file.pdf')
        mock_get_document_text.assert_called_once_with(self.mock_reader)
        mock_get_sentences.assert_called_once()


    @patch("docproc.extract_emails")
    @patch("docproc.extract_dates")
    def test_extract_features_from_sentence(self, mock_extract_dates, mock_extract_emails):
        mock_extract_dates.return_value = ["January 1, 2024"]
        mock_extract_emails.return_value = ["test@example.com"]

        sentence_text = "This is a sentence with an email test@example.com and a date January 1, 2024."
        features = dp.extract_features_from_sentence(self.nlp, sentence_text)

        self.assertEqual(features, {
            "dates": ["January 1, 2024"],
            "email_addresses": ["test@example.com"],
            "text": sentence_text
        })

        mock_extract_dates.assert_called_once_with(self.nlp, sentence_text)
        mock_extract_emails.assert_called_once_with(self.nlp, sentence_text)


if __name__ == "__main__":
    unittest.main()
