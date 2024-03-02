import streamlit as st
import pandas as pd
import requests
#st.set_page_config(layout="wide")

# Header
st.header(':orange[Law Bot statics] :sunflower:', divider='rainbow')

# Initial perpertual parameter
## Initial chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

## Initial button select 
if "in_brief" not in st.session_state:
    st.session_state.in_brief = True

# url = 'https://dbapi-mtfui6kmqq-uc.a.run.app'
url = 'https://dbapi-mtfui6kmqq-de.a.run.app'
# url = 'https://24c8-2402-7500-578-5705-c8a0-fb52-7516-59ed.ngrok-free.app'

# Read data
detailed_history = requests.get(f'{url}/detailed_history').json()
history = requests.get(f'{url}/history', params={'userId':'ALL'}).json()

############################################ Show statics ############################################
if st.session_state['in_brief']:
    st.subheader('Brief summary')

    # Select Button
    if st.button('Personal'):
        st.session_state['in_brief'] = False
        st.rerun()

    # Basic statics
    col1, col2, col3 = st.columns(3)
    col1.metric("使用者數量", "57位", "8位")
    col2.metric("訪問次數", "182次", "-16次")
    col3.metric("總token金額", "USD 2.1", "0.09")
        
    # Token price cost
    st.bar_chart({key:sum(item['total_price'] for item in value) for key,value in detailed_history.items()})    


    # 處理資料成適合的格式
    data = []
    cumulative_total = 0
    for key, value in detailed_history.items():
        total_price = sum(item['total_price'] for item in value)
        cumulative_total += total_price
        # 使用有效的日期時間格式替換無效的鍵
        try:
            date = pd.to_datetime(key)
        except ValueError:
            date = pd.Timestamp('now')
        data.append({'Date': date, 'Cumulative_Total': cumulative_total})

    df = pd.DataFrame(data)

    # # 將日期設定為索引
    # df.set_index('Date', inplace=True)

    # # 選擇月份
    # selected_month = st.slider("Select month", 1, 12, 1)

    # # 篩選數據
    # df_current_month = df[df.index.month == selected_month]

    # # 繪製圖表
    # st.line_chart(df_current_month)

    # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    # st.line_chart(chart_data)

############################################ Show personal statics ############################################
else:
    st.subheader('Personal summary')

    # Select Button
    if st.button('Brief'):
        st.session_state['in_brief'] = True
        st.rerun()

    # SelectBox
    option = st.selectbox(
        '',
        list(history.keys()))

    ## Add messages
    ### Reset messages
    st.session_state.messages = []
    messages = [every_chat.split(': ')[1].strip() for every_chat in history[option].split('\n')]
    for message_index in range(0, len(messages), 2):
        st.session_state.messages.append({"role": "user", "content": messages[message_index]})
        st.session_state.messages.append({"role": "assistant", "content": messages[message_index+1]})

    ## Display chat messages in side bar
    with st.sidebar:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    ## Display chat detailed messages in page
    st.dataframe(pd.DataFrame(detailed_history[option]))

