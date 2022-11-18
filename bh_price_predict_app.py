import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# shap library provides the understanding behind the prediction
import shap
import pickle

# in Python the base64 module is used to encode and decode data
# at first the strings are converted into byte-like objects and then encoded using the base64 module
import base64

# disabling the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)


# configuring the settings of the page
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon=":house:",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# title and header
st.title('House Price Prediction')
st.write('This app predicts the prices for houses and residential plots based out of Boston in USA')
st.write('---')

# loading the Boston House Price dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=['MEDV'])

# ----------------------------------------------------------------------------------------------------------------------------
# setting up the cover page for our application


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
    # we use the base64.b64encode() function to encode the base64 bytes
    # now to get a string out of these bytes we use decode() method


def add_image(cover_image):
    bin_str = get_base64(cover_image)
    bg_img = '''
    <style>.stApp{
    background-image: url("data:image/png;base64,%s");
    background-attachment: fixed;
    background-size: cover
    }
    </style>
    ''' % bin_str
    st.markdown(bg_img, unsafe_allow_html=True)
    # background-size: cover --> meaning the background image will cover the entire element
    # background-attachment: fixed --> meaning entire element is always covered with no stretching (the image will retain its original proportion)
    # unsafe_allow_html=False --> to ensure the HTML tags in the body are neither escaped nor treated as mere text


add_image('./Boston House Pricing/Images/image_7.jpg')

# ----------------------------------------------------------------------------------------------------------------------------

# sidebar to specify input parameters
# header and title for the sidebar
st.sidebar.subheader('Please specify the input parameters')


def user_input_features():
    # for each feature we are paaing 3 parameters as -
    # - minimum value: lower limit of the slider
    # - maximum value: upper limit of the slider
    # - mean value: default value when we use the sidebar panel

    # min(), max() & mean() functions are retruning values of int64 datatype instead of int
    # while our program expects these values in float owing to the continuous values in our dataset
    # so we are converting these values to respective 'float' datatype while passing them as parameters

    CRIM = st.sidebar.slider('per capita crime rate', float(X.CRIM.min()),
                             float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('proportion of residential land zoned', float(X.ZN.min()),
                           float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('proportion of non-retail business acres', float(X.INDUS.min()),
                              float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('bounded by Charles River (Yes/No)', float(X.CHAS.min()),
                             float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('nitric oxides concentration (parts per 10 million)', float(X.NOX.min()),
                            float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('average number of rooms', float(X.RM.min()),
                           float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('proportion of owner-occupied units', float(X.AGE.min()),
                            float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('distance to five employment centres in Boston', float(X.DIS.min()),
                            float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('index of accessibility to radial highways', float(X.RAD.min()),
                            float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('full-value property-tax rate per $10000', float(X.TAX.min()),
                            float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('pupil-teacher ratio', float(X.PTRATIO.min()),
                                float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('proportion of people of African-American descent', float(X.B.min()),
                          float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('percentage of lower status of the population', float(X.LSTAT.min()),
                              float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=['Values'])
    return features


df = user_input_features()


# print the specified input parameters
st.subheader('Input parameters from the user')
st.dataframe(df)
st.write('---')


# unpickle the 'xgb_model.pkl' file
model = pickle.load(open('./Boston House Pricing/xgb_model.pkl', 'rb'))
prediction = pd.DataFrame(model.predict(df.values) *
                          1000, index=['Value'], columns=['Median Price (in USD)'])


# displaying the predicted price
st.subheader('Predicted Median Price')
st.write(prediction)
output = 'Predicted median price for the parameters as specified by the user: $ {}'.format(
    float(prediction.values))
st.markdown(output)
st.write('---')


# explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
st.subheader('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('')

plt.title('Feature importance based on SHAP values (Bar Indicator)')
shap.summary_plot(shap_values, X, plot_type='bar')
st.pyplot(bbox_inches='tight')
