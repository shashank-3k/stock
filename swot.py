import streamlit as st
#st.header('Swot Analysis')

def main():
    st.title("Swot Analysis")
    stock = st.selectbox("Select a stock", ["AAPL", "AMZN", "MSFT", "GOOG", "TSLA", "NVDA", "COST", "ADBE", "WMT"])
    if stock=="AAPL":   
        st.subheader('Strength:')
        st.text('Rising Delivery Percentage Compared to Previous Day and Month, Strong Volumes')
        st.text('High Revenue and Profit Growth with High Return on Capital Deployed (ROCE) and Low PE ratio')
        
        st.title('Weakness:')
        st.text('High PE with Negative ROE')
        st.subheader('Opportunities:')
        st.text('RSI indicating price strength')
        st.subheader('Threats:')
        st.text('Red Flag: Resignation of Top Management')
        
        
    elif stock=="AMZN":   
        st.subheader('Strength:')
        st.text('Amazon has instant brand awareness')
        st.text('Amazon is a market leader')
        st.text('Amazon offers a minimum wage of $15 per hour')
        
        st.subheader('Weakness:')
        st.text('Amazon loses revenue in some areas, including shipping')
        st.text('Fraudulent reviews of Amazon products are a problem')
        
        st.subheader('Opportunities:')
        st.text('Amazon’s ecosystem can be improved')
        st.text('Utilizing self-driving vehicles would save on driver costs')
        
        st.subheader('Threats:')
        st.text('50% of Amazon’s product distribution is handled by outsourced supplier')
        
        
    elif stock == "GOOG":
        st.subheader('Strength:')
        st.text('Excellent acquisition capabilities')
        st.text('High research and development (R&D) expenditure resulting in one of the fastest growing and strongest patent portfolios')

        st.subheader('Weakness:')
        st.text('Poor user experience of the Android OS due to increasing fragmentation')
        st.text('Overdependence on revenue from advertising')

        st.subheader('Opportunities:')
        st.text('Edge computing market size will grow to US$15.7 billion by 2025')
        st.text('Growing market for subscription-based video on demand services')

        st.subheader('Threats:')
        st.text('Growing privacy concerns and the possibility of data breaches')

        
        
    elif stock=="MSFT":
        st.subheader('Strength:')
        st.text('Company has been maintaining a healthy dividend payout of 58.8%')
        st.text('Company has a good return on equity (ROE) track record: 3 Years ROE 29.4%')

        st.subheader('Weakness:')
        st.text('Stock is trading at 7.17 times its book value')
        st.text('Company might be capitalizing the interest cost')

        st.subheader('Opportunities:')
        st.text('Brokers upgraded recommendation or target price in the past three months')
        st.text('Insiders bought stocks')

        st.subheader('Threats:')
        st.text('No threats found')
        
    elif stock=="NVDA":
        st.subheader('Strength:')
        st.text('Leveraging brand recognition in new segments')
        st.text('NVIDIA is recognized for its cutting-edge GPU technology')
        st.text('NVIDIA has demonstrated strong financial performance with consistent revenue growth and profitability over the years')

        st.subheader('Weakness:')
        st.text('NVIDIA revenue heavily relies on the gaming and data center segments, making it vulnerable to fluctuations in these markets')
        st.text('Exposure to cyclical demand patterns')
        st.text('Potential for intellectual property infringement')

        st.subheader('Opportunities:')
        st.text('Growth of the data center market')
        st.text('Increasing adoption of AI')
        st.text('Development of new products and services')

        st.subheader('Threats:')
        st.text('Trade wars')
        st.text('Changes in technology')
            
    elif stock=="TSLA":
        st.subheader('Strength:')
        st.text('Tesla has a strong brand recognition, which gives the company a competitive advantage')
        st.text('Innovative technology')
        st.text('Strong financial performance')
        
        st.subheader('Weakness:')
        st.text('Tesla is facing a shortage of skilled labor, which could slow down the companys production and make it difficult to meet demand')
        st.text('Tesla is facing regulatory challenges in some countries, which could make it difficult for the company to expand into new markets')
        
        st.subheader('Opportunities:')
        st.text('Tesla has the opportunity to acquire new businesses, which could give the company access to new technologies or markets')
        st.text('The electric vehicle market is growing rapidly, which gives Tesla the opportunity to grow its business')
        
        st.subheader('Threats:')
        st.text('No threats found')
        
        
    elif stock=="WMT":
        st.subheader('Strength:')
        st.text('Strong brand recognition and reputation')
        st.text('Large and loyal customer base')
        st.text('Efficient supply chain')
        
        st.subheader('Weakness:')
        st.text('Increasing competition from online retailers')
        st.text('Reliance on brick-and-mortar stores')
        
        st.subheader('Opportunities:')
        st.text('Expansion into new markets')
        
        st.subheader('Threats:')
        st.text('Economic downturn')
        st.text('Changes in consumer behavior')
        
    elif stock=="ADBE":
        st.subheader('Strength:')
        st.text('Strong brand recognition and reputation')
        st.text('Large and loyal customer base')
        st.text('Efficient supply chain')
        
        st.subheader('Weakness:')
        st.text('Increasing competition from online retailers')
        st.text('Reliance on brick-and-mortar stores')
        
        st.subheader('Opportunities:')
        st.text('Expansion into new markets')
        
        st.subheader('Threats')
        st.text('no threats found')
    
    elif stock=="COST":
        st.subheader('Strength:')
        st.text('COST has a strong brand recognition, which gives the company a competitive advantage')
        st.text('strong customer loyalty, which makes it less likely that customers will switch to other brands')
        st.text('experienced management team, which gives the company the expertise to navigate the challenges it faces')
        
        st.subheader('Weakness:')
        st.text('high cost of production, which could make it difficult for the company to compete with lower-cost rivals.')
        st.text('faces increasing competition from online retailers')
        
        st.subheader('Opportunities:')
        st.text('Opportunity to expand into new markets, such as China and Europe')
        st.text('The e-commerce market is growing rapidly, which gives COST the opportunity to grow its business.')
        
        st.subheader('Threats')
        st.text('New entrants to the market could pose a threat to COST market share')
        st.text('Changes in regulations')
        
        
    
        
    
        
        
        # with open(r'C:\Users\pande\Downloads\tsla.txt', encoding='iso-8859-1') as f:
        # state_of_the_union = f.read()
if __name__ == '__main__':
    main()

