import streamlit as st
import time
#st.header('Swot Analysis')

def swot_main():
    st.title("Swot Analysis")
    stock = st.selectbox("Select a stock", ["AAPL", "AMZN", "MSFT", "GOOG", "TSLA", "NVDA", "COST", "ADBE"])
    if st.button("Submit"):
        with st.spinner("Loading..."):
            time.sleep(2)
            if stock=="AAPL":   
                st.subheader('Strength:')
                st.text('Rising Delivery Percentage Compared to Previous Day and Month, Strong Volumes', style={'overflow': 'hidden'})
                st.text('High Revenue and Profit Growth with High Return on Capital Deployed (ROCE) and Low PE ratio', style={'overflow': 'hidden'})
                
                st.subheader('Weakness:')
                st.text('High PE with Negative ROE', style={'overflow': 'hidden'})
                st.subheader('Opportunities:')
                st.text('RSI indicating price strength', style={'overflow': 'hidden'})
                st.subheader('Threats:')
                st.text('Red Flag: Resignation of Top Management', style={'overflow': 'hidden'})
                
                
            elif stock=="AMZN":   
                st.subheader('Strength:')
                st.text('Amazon has instant brand awareness', style={'overflow': 'hidden'})
                st.text('Amazon is a market leader', style={'overflow': 'hidden'})
                st.text('Amazon offers a minimum wage of $15 per hour', style={'overflow': 'hidden'})
                
                st.subheader('Weakness:')
                st.text('Amazon loses revenue in some areas, including shipping', style={'overflow': 'hidden'})
                st.text('Fraudulent reviews of Amazon products are a problem', style={'overflow': 'hidden'})
                
                st.subheader('Opportunities:')
                st.text('Amazon’s ecosystem can be improved', style={'overflow': 'hidden'})
                st.text('Utilizing self-driving vehicles would save on driver costs', style={'overflow': 'hidden'})
                
                st.subheader('Threats:')
                st.text('50% of Amazon’s product distribution is handled by outsourced supplier', style={'overflow': 'hidden'})
                
                
            elif stock == "GOOG":
                st.subheader('Strength:')
                st.text('Excellent acquisition capabilities', style={'overflow': 'hidden'})
                st.text('High research and development (R&D) expenditure resulting in one of the fastest growing and strongest patent portfolios', style={'overflow': 'hidden'})

                st.subheader('Weakness:')
                st.text('Poor user experience of the Android OS due to increasing fragmentation', style={'overflow': 'hidden'})
                st.text('Overdependence on revenue from advertising', style={'overflow': 'hidden'})

                st.subheader('Opportunities:')
                st.text('Edge computing market size will grow to US$15.7 billion by 2025', style={'overflow': 'hidden'})
                st.text('Growing market for subscription-based video on demand services', style={'overflow': 'hidden'})

                st.subheader('Threats:')
                st.text('Growing privacy concerns and the possibility of data breaches', style={'overflow': 'hidden'})

                
                
            elif stock=="MSFT":
                st.subheader('Strength:')
                st.text('Company has been maintaining a healthy dividend payout of 58.8%', style={'overflow': 'hidden'})
                st.text('Company has a good return on equity (ROE) track record: 3 Years ROE 29.4%', style={'overflow': 'hidden'})

                st.subheader('Weakness:')
                st.text('Stock is trading at 7.17 times its book value', style={'overflow': 'hidden'})
                st.text('Company might be capitalizing the interest cost', style={'overflow': 'hidden'})

                st.subheader('Opportunities:')
                st.text('Brokers upgraded recommendation or target price in the past three months', style={'overflow': 'hidden'})
                st.text('Insiders bought stocks', style={'overflow': 'hidden'})

                st.subheader('Threats:')
                st.text('No threats found', style={'overflow': 'hidden'})
                
            elif stock=="NVDA":
                st.subheader('Strength:')
                st.text('Leveraging brand recognition in new segments', style={'overflow': 'hidden'})
                st.text('NVIDIA is recognized for its cutting-edge GPU technology', style={'overflow': 'hidden'})
                st.text('NVIDIA has demonstrated strong financial performance with consistent revenue growth and profitability over the years', style={'overflow': 'hidden'})

                st.subheader('Weakness:')
                st.text('NVIDIA revenue heavily relies on the gaming and data center segments, making it vulnerable to fluctuations in these markets', style={'overflow': 'hidden'})
                st.text('Exposure to cyclical demand patterns', style={'overflow': 'hidden'})
                st.text('Potential for intellectual property infringement', style={'overflow': 'hidden'})

                st.subheader('Opportunities:')
                st.text('Growth of the data center market', style={'overflow': 'hidden'})
                st.text('Increasing adoption of AI', style={'overflow': 'hidden'})
                st.text('Development of new products and services', style={'overflow': 'hidden'})

                st.subheader('Threats:')
                st.text('Trade wars', style={'overflow': 'hidden'})
                st.text('Changes in technology', style={'overflow': 'hidden'})
                    
            elif stock=="TSLA":
                st.subheader('Strength:')
                st.text('Tesla has a strong brand recognition, which gives the company a competitive advantage', style={'overflow': 'hidden'})
                st.text('Innovative technology', style={'overflow': 'hidden'})
                st.text('Strong financial performance', style={'overflow': 'hidden'})
                
                st.subheader('Weakness:')
                st.text('Tesla is facing a shortage of skilled labor, which could slow down the companys production and make it difficult to meet demand', style={'overflow': 'hidden'})
                st.text('Tesla is facing regulatory challenges in some countries, which could make it difficult for the company to expand into new markets', style={'overflow': 'hidden'})
                
                st.subheader('Opportunities:')
                st.text('Tesla has the opportunity to acquire new businesses, which could give the company access to new technologies or markets', style={'overflow': 'hidden'})
                st.text('The electric vehicle market is growing rapidly, which gives Tesla the opportunity to grow its business', style={'overflow': 'hidden'})
                
                st.subheader('Threats:')
                st.text('No threats found')
                
                
                
            elif stock=="ADBE":
                st.subheader('Strength:')
                st.text('Strong brand recognition and reputation', style={'overflow': 'hidden'})
                st.text('Large and loyal customer base', style={'overflow': 'hidden'})
                st.text('Efficient supply chain', style={'overflow': 'hidden'})
                
                st.subheader('Weakness:')
                st.text('Increasing competition from online retailers', style={'overflow': 'hidden'})
                st.text('Reliance on brick-and-mortar stores', style={'overflow': 'hidden'})
                
                st.subheader('Opportunities:')
                st.text('Expansion into new markets', style={'overflow': 'hidden'})
                
                st.subheader('Threats')
                st.text('no threats found', style={'overflow': 'hidden'})
            
            elif stock=="COST":
                st.subheader('Strength:')
                st.text('COST has a strong brand recognition, which gives the company a competitive advantage', style={'overflow': 'hidden'})
                st.text('strong customer loyalty, which makes it less likely that customers will switch to other brands', style={'overflow': 'hidden'})
                st.text('experienced management team, which gives the company the expertise to navigate the challenges it faces', style={'overflow': 'hidden'})
                
                st.subheader('Weakness:')
                st.text('high cost of production, which could make it difficult for the company to compete with lower-cost rivals.', style={'overflow': 'hidden'})
                st.text('faces increasing competition from online retailers', style={'overflow': 'hidden'})
                
                st.subheader('Opportunities:')
                st.text('Opportunity to expand into new markets, such as China and Europe', style={'overflow': 'hidden'})
                st.text('The e-commerce market is growing rapidly, which gives COST the opportunity to grow its business.', style={'overflow': 'hidden'})
                
                st.subheader('Threats')
                st.text('New entrants to the market could pose a threat to COST market share', style={'overflow': 'hidden'})
                st.text('Changes in regulations', style={'overflow': 'hidden'})
        
        
    
        
    
        
        
        # with open(r'C:\Users\pande\Downloads\tsla.txt', encoding='iso-8859-1') as f:
        # state_of_the_union = f.read()
if __name__ == '__main__':
    swot_main()

