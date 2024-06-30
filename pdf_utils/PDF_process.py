from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import PyPDF2, nltk
from pine_upload import upsert_data

def get_compressed_context_from_hf_api(pdf_path):
    '''
    It compresses the text from pdf using the NLP model. This also includes breaking the text into chunks and summarizing them
    @param pdf_path = path to pdf file to compress and upsert it into DB
    
    @return Summarized text of pdf file
    '''
    
    concat_temp_summary = ""
    summarizer = LexRankSummarizer()

    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for i in range(4,25,2):
            try :
                input_text =  pdf_reader.pages[i].extract_text() + pdf_reader.pages[i+1].extract_text()
                print(f"\nThis is length of text of {i} and {i+1} pages :",len(input_text))
            except IndexError:
                input_text = pdf_reader.pages[i].extract_text()
                print(f'Only one page found at index {i}. Length: {len(input_text)}')

            except Exception as e:
                print(f'Error concatenating text from pages {i} and {i + 1}: {e}')
                continue
            
            parser = PlaintextParser.from_string(input_text, Tokenizer("english"))

            # Customize your summary's length
            summary = summarizer(parser.document, sentences_count=80)


            temp_summary = ' '.join(str(text) for text in summary)
            print("Len of summary = ",len(temp_summary))

            concat_temp_summary = concat_temp_summary + temp_summary
            
    return concat_temp_summary


if __name__ == '__main__':

    cname = "uber"
    pdf_path = f"Pdf processing/data/{cname}.pdf"

    text = get_compressed_context_from_hf_api(pdf_path)
    print(len(text))

    upsert_data(cname, text)
    

    

    



