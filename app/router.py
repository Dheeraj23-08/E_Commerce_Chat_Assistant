from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(
    name="sentence-transformers/all-MiniLM-L6-v2"
)

faq = Route(
    name='faq',
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
    ]
)

sql = Route(
    name='sql',
    utterances=[
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of puma running shoes?",
    ]
)
small_talk = Route(
    name='small_talk',
    utterances=[
        "How are you?",
        "Hlo",
        "What is your name?",
        "Are you a robot?",
        "What are you?",
        "What do you do?",
        "Who created you?",
        "Where are you from?",
        "Can you help me?",
        "What can you do?",
        "How old are you?",
        "Do you sleep?",
        "Do you have feelings?",
        "Are you human?",
        "Where do you live?",
        "Do you like me?",
        "Can we be friends?",
        "Tell me a joke.",
        "What’s your favorite color?",
        "Do you have a family?",
        "Who is your best friend?",
        "What’s the weather like?",
        "Do you know me?",
        "What’s your favorite food?",
        "Can you sing?",
        "Can you dance?",
        "What’s your favorite movie?",
        "Do you believe in God?",
        "Can you think?",
        "Do you have emotions?",
        "Are you intelligent?",
        "Can you tell me a story?",
        "Do you love me?",
        "What’s your purpose?",
        "How smart are you?",
        "Can you learn?",
        "Are you always online?",
        "How do you work?",
        "Do you like humans?",
        "Can you speak other languages?",
        "What day is it today?",
        "What time is it?",
        "Who made you?",
        "Are you conscious?",
        "What’s your favorite song?",
        "Can you feel pain?",
        "Do you get tired?",
        "Can you dream?",
        "Do you have a name?",
        "Why were you created?",
        "Can I teach you?",
    ]
)



router= SemanticRouter(routes=[faq,sql,small_talk],encoder=encoder,auto_sync="local")
test_data = [
    # FAQ-related queries
    ("What is the return policy of the products?", "faq"),
    ("How long does it take to process a refund?", "faq"),
    ("What payment methods are accepted?", "faq"),
    ("How can I track my order?", "faq"),
    ("Do I get discount with the HDFC credit card?", "faq"),
    ("I received a damaged product, what do I do?", "faq"),
    ("My item is defective — how do I get a replacement?", "faq"),
    ("I want a refund for a faulty product", "faq"),
    ("Is there a warranty or replacement for defective items?", "faq"),
    ("The product I got is broken, how to return it?", "faq"),
    ("Can I return an item bought during sale?", "faq"),
    ("How long will my refund take to reflect?", "faq"),
    ("How do I cancel my order?", "faq"),
    ("Do you offer free returns?", "faq"),
    ("Who pays for return shipping?", "faq"),

    # SQL / search-related queries
    ("I want to buy Nike shoes that have 50% discount", "sql"),
    ("Are there any shoes under Rs. 3000?", "sql"),
    ("Do you have formal shoes in size 9?", "sql"),
    ("Are there any Puma shoes on sale?", "sql"),
    ("What is the price of Puma running shoes?", "sql"),
    ("Show me red running shoes for men below 4000", "sql"),
    ("Find women's black heels size 7 under 2000", "sql"),
    ("Any offers on Apple AirPods?", "sql"),
    ("Search Samsung mobiles with 8GB RAM and 128GB storage", "sql"),
    ("I want waterproof jackets in XL", "sql"),
    ("Find Bluetooth speakers under Rs. 1500", "sql"),
    ("Show me laptops with i5 and 16GB RAM", "sql"),
    ("Are there any deals on refrigerators?", "sql"),
    ("Show me gold-plated earrings under 1000", "sql"),
    ("Any cashback offers with SBI credit card?", "sql"),

    # Small talk queries
    ("How are you?", "small_talk"),
    ("What is your name?", "small_talk"),
    ("Are you a robot?", "small_talk"),
    ("What are you?", "small_talk"),
    ("Who created you?", "small_talk"),
    ("Where are you from?", "small_talk"),
    ("Can you help me?", "small_talk"),
    ("What can you do?", "small_talk"),
    ("How old are you?", "small_talk"),
    ("Do you sleep?", "small_talk"),
    ("Do you have feelings?", "small_talk"),
    ("Are you human?", "small_talk"),
    ("Do you like me?", "small_talk"),
    ("Can we be friends?", "small_talk"),
    ("Tell me a joke.", "small_talk"),
    ("What’s your favorite color?", "small_talk"),
    ("Do you have a family?", "small_talk"),
    ("Who is your best friend?", "small_talk"),
    ("Do you know me?", "small_talk"),
    ("Can you sing?", "small_talk"),
    ("Can you dance?", "small_talk"),
    ("What’s your favorite movie?", "small_talk"),
    ("Do you believe in God?", "small_talk"),
    ("Can you tell me a story?", "small_talk"),
    ("Do you love me?", "small_talk"),
    ("What’s your purpose?", "small_talk"),
    ("Can you learn?", "small_talk"),
    ("Are you always online?", "small_talk"),
    ("Do you like humans?", "small_talk"),
    ("What day is it today?", "small_talk"),
    ("What time is it?", "small_talk"),
    ("Can you feel pain?", "small_talk"),
    ("Do you get tired?", "small_talk"),
    ("Can you dream?", "small_talk"),
    ("Why were you created?", "small_talk"),
    ("Can I teach you?", "small_talk"),
]


X, y = zip(*test_data)
router.fit(X=X, y=y)
route_thresholds = router.get_thresholds()
print("Updated route thresholds:", route_thresholds)
accuracy = router.evaluate(X=X, y=y)
print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__=="__main__":
    print(router("What is your policy on defective products?").name)
    print(router("Pink Puma shoes in price range 5000 to 1000").name)
    print(router("what is your name?").name)
