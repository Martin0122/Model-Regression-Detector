from src.classifier import classify_email
from src.services.load_configs import load_prompt_config
import asyncio

TEST_BILLING = """
Subject: Cancel subscription renewal

Hello,
I would like to cancel my subscription before the next billing cycle. Please confirm that I will not be charged again.

Best regards.
"""
TEST_ACCOUNT = """
Subject: Cannot reset my password

Hello,
I forgot my password and attempted to use the password reset link, but I never received the email. Could you help me regain access to my account?

Thank you.
"""
TEST_TECHNICAL = """
Subject: Unable to connect to the API endpoint

Hello Support Team,
I am trying to connect my application to your API, but I keep receiving a 500 internal server error whenever I send a request. I have verified my API key and checked my configuration, but the issue still persists. Could you help me troubleshoot this?
"""
TEST_GENERAL = """
Subject: Request for product information

Hi,
Could you provide more information about your service and how it works? I would like to understand how your platform can help my team manage our workflow.

Thank you.
"""


async def run_tests():
    config = load_prompt_config("./prompts/v1_classifier.yaml")

    print("Billing Test:", await classify_email(TEST_BILLING, config))
    print("Account Test:", await classify_email(TEST_ACCOUNT, config))
    print("Technical Test:", await classify_email(TEST_TECHNICAL, config))
    print("General Test:", await classify_email(TEST_GENERAL, config))


if __name__ == "__main__":
    asyncio.run(run_tests())
