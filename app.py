from io import BytesIO

import numpy
import streamlit as st
import pandas as pd
import openpyxl
import parameters as p
from xlsxwriter import Workbook
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from sentence_transformers import SentenceTransformer, util

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.000'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


if 'file' not in st.session_state:
    st.session_state['file'] = 0

if 'embeddings_list' not in st.session_state:
	st.session_state.embeddings_list = []

def main():
    
    model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
    st.set_page_config(layout="wide")
    st.title('Similitud de textos')
    uploaded_file = st.file_uploader("Choose a file")
    if st.session_state.file != uploaded_file:
        st.session_state.file = uploaded_file
        st.session_state.embeddings_list = []
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, skiprows=p.ROWS_BEFORE_HEADER)
        #with col1:
        input_texto = st.text_input('Ingrese texto a comparar')
        result = st.button('Buscar')
        df_result = pd.DataFrame(columns=('cosine_score', p.TEXT_COLUMN_NAME))
        if result:
            seed = input_texto
            embeddings1 = model.encode(seed)
            with st.spinner('Buscando sentencias similares...'):
                exceeds = len(df.index) > p.MAX_DATA
                iterator = range(p.MAX_DATA) if exceeds else df.index
                if len(st.session_state.embeddings_list) == 0:
                    st.write('No se encontro caché, generando nuevo.')
                    for i in iterator:
                        compare = df[p.TEXT_COLUMN_NAME][i]
                        embeddings2 = model.encode(compare)
                        st.session_state.embeddings_list.append(embeddings2)                            
                        cosine_scores = util.cos_sim(embeddings1, embeddings2)
                        cosine = numpy.around(cosine_scores.numpy()[0], decimals=3)
                        print(cosine)
                        comp = {'cosine_score': cosine, p.TEXT_COLUMN_NAME: compare}
                        df_result = df_result.append(comp, ignore_index=True)
                    print(len(st.session_state.embeddings_list))
                else:
                    st.write('Utilizando caché almacenado.')
                    for i in iterator:
                        compare = df[p.TEXT_COLUMN_NAME][i]
                        embeddings2 = st.session_state.embeddings_list[i]                            
                        cosine_scores = util.cos_sim(embeddings1, embeddings2)
                        cosine = numpy.around(cosine_scores.numpy()[0], decimals=3)
                        print(cosine)
                        comp = {'cosine_score': cosine, p.TEXT_COLUMN_NAME: compare}
                        df_result = df_result.append(comp, ignore_index=True)
            df_result = df_result.sort_values(['cosine_score'], ascending=[False])
            #    with col2:
            st.header("Resultados")
            gb2 = GridOptionsBuilder.from_dataframe(df_result)
            gb2.configure_pagination(paginationAutoPageSize=True)  # Add pagination
            gb2.configure_side_bar()  # Add a sidebar
            gridOptions2 = gb2.build()

            grid_result = AgGrid(
                df_result,
                gridOptions=gridOptions2,
                data_return_mode='AS_INPUT',
                update_mode='MODEL_CHANGED',
                fit_columns_on_grid_load=False,
                enable_enterprise_modules=False,
                height=350,
                width='100%',
                reload_data=True
            )
            df_xlsx = to_excel(df_result)
            st.download_button(label='📥 Resultados',
                                data=df_xlsx,
                                file_name='resultados.xlsx')


if __name__ == "__main__":
    main()
