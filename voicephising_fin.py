
### voice to text ###

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# import torch
# import librosa
# import numpy as np
# import soundfile as sf
# from scipy.io import wavfile
# from IPython.display import Audio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# file_name = "HSBC_Scam_call.wav"
# Audio(file_name)

# data = wavfile.read(file_name)
# framerate = data[0]
# sounddata = data[1]
# time = np.arange(0,len(sounddata))/framerate
# input_audio, _ = librosa.load(file_name, sr=16000)
# input_values = tokenizer(input_audio, return_tensors="pt").input_values
# logits = model(input_values).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = tokenizer.batch_decode(predicted_ids)[0]


# print('transcription :', transcription)

# -------------------------------------------------------------------------------#
### 보이스피싱에 반드시 등장하는 단어들을 필터로 만들어서, 이 단어가 통화시 발견되면 보이스피싱일 가능성이 있다는 로직 ##
# keywords extraction

# 스크립트
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.tokenize import sent_tokenize

class VP:

    def __init__(self) -> None:
        pass
        
    def extKeywords(self, scripts):
        n_gram_range = (1, 1)
        stop_words = "english"
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([scripts])
        candidates = count.get_feature_names()


        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embedding = model.encode([scripts])
        candidate_embeddings = model.encode(candidates)

        from sklearn.metrics.pairwise import cosine_similarity

        top_n = 40
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords_re = [candidates[index] for index in distances.argsort()[0][-top_n:]]
        #print("keywords :", keywords_re)
        return keywords_re

    def vp_lexcon(self, scripts):
        vp = VP()
        ext_kws = vp.extKeywords(scripts)
        return ext_kws


    ## main operation ##    
    def vali_voice_phi(self, vp_comp_keywords , input_sent):
        vp = VP()
        li_comp_sent = input_sent
        getVoiceTextSet = vp.extKeywords(li_comp_sent)
        # 단어 포함여부 개별 검사
        matching_keywords = []
        for i in getVoiceTextSet:

            if i in vp_comp_keywords:
                #print("found!")
                matching_keywords.append(i)
            else:
                #print("Not found!")
                pass


        result = len(matching_keywords) / len(getVoiceTextSet) * 100
        print('Voice Phishing Probability by keywords :', result)
        return result


    ### 보이스피싱 문장과 기존의 수집된 보이스피싱관련 문장들과의 유사도록 분석하는 방법 ###
    def simli(self,  inp_scripts):
        scripts = inp_scripts
        result_ = sent_tokenize(scripts)

        li_re = []
        for i in result_:

            re_ = i.strip("R. Williams:")
            re_ = re_.strip("Williams:")
            re_ = re_.strip("S. Parker:")
            re_ = re_.strip("Parker:")
            li_re.append(re_)

        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('bert-base-nli-mean-tokens')

        li_comp_sent = sent_tokenize(scripts)


        simility_resut_list = []
        for sent in li_re:
            for sent_ in li_comp_sent:
                sen_input =[sent, sent_]
                #Encoding:
                sen_embeddings = model.encode(sen_input)
                #let's calculate cosine similarity for sentence 0:
                sim_re = cosine_similarity(
                    [sen_embeddings[0]],
                    sen_embeddings[1:]
                )

                # 문장 유사도 비교 결과 input_sent_A. vs. input_sent_B
                cal_sim = sim_re[0][0]
                simility_resut_list.append(cal_sim)


        # 보정을 위해서 최대, 최소값 삭제
        min_value = min(simility_resut_list)
        max_value = max(simility_resut_list)

        simility_resut_list.remove(min_value)
        simility_resut_list.remove(max_value)

        #print("최대 최소값 삭제 확인: " , simility_resut_list)


        # 개별 비교 분석 결과 평균값 산출하기
        import numpy as np
        a = np.array(simility_resut_list)
        AVG = np.mean(a) * 100
        return AVG




# 보이스피싱인지 알아보려는 비교 스크립트
## Voice Phishing Example
li_comp_sent1 = """Hello, this is the IRS. What is your name and your address? I am calling to inform you that the IRS has filed a lawsuit against you. You have been filed for tax evasion. You owe us 2,100 dollars if you don’t want to go to court. Please pay through gift cards or wire transfer."""
li_comp_sent2 = """Good evening, this is the Internal Revenue Service. In order to confirm your identity, please give me your full name and address. Yes, Ashley, you are being called to court due to tax evasion.  You currently have 214,000 dollars of taxes unpaid. If this amount remains unpaid, then you will be taken to court. However, if you prefer avoiding the long court process, please make a wire transfer of 214,000 to this bank account."""
li_comp_sent3 = """Hello, this is amazon. You have called because someone has made a purchase on your account that you would like to cancel. If you want to cancel and get a refund, you must pay us with gift cards first to verify your identity. If you do not buy a gift card, we can not verify your identity, and you will not be able to receive your money. This is a very necessary process, please trust us."""
li_comp_sent4 = """Thank you for calling amazon, what may I help you with? Would you like to cancel the purchase of the table lamp? Ok, we can do that for you. Please call out your full name, address, and credit card number. This is just to verify your identity. Now, please send a gift card to us to verify your identity."""
li_comp_sent5 = """Hello, this is your credit card company. We believe that someone has hacked your account and stolen money. In order to confirm that you are the owner of this credit card, please call out your full name, address, and credit card number."""
li_comp_sent6 = """Hello, this is Wells Fargo customer service, how may I help you? Just to confirm, you received our email that someone has stolen your credit card number and is making unauthorized purchases? Ok, got it. In order for me to confirm your identity, please call out your full name and credit card number. Ok, I see you on our system, let me get this situation sorted for you."""
li_comp_sent7 = """Hello, this is Leaders hagwon. We are calling to confirm that you signed your daughter up for this winter’s SAT session. Yes, thank you for your confirmation. Please pay the tuition, 4,000 dollars right now by wire transfer. Or, you could call out your credit card number right now. Either way, if the money is not sent by now, your child will not be able to attend the session."""

## Voice Phishing Non-Example:
li_comp_sent8 = """Hello, is this Ashley Tisdale? Yes, we’re calling to let you know that a payment for your monthly gym membership is due tomorrow. Please make sure you stop by and make your payment!"""
li_comp_sent9 = """Hi Lauren, what do you want for your Christmas present? Would you like a gift card or a bought present? A gift card? Ok, then I will give you a gift card for your Christmas present."""
li_comp_sent10 = """Hi Gabe, I’m so sorry but I think I lost my credit card! Could you please check if my credit card is accidentally in your wallet. It has my full name, Steve Jobs on it. Is it there? Thank you so much"""
li_comp_sent11 = """Hi mom, could you tell me the address of our house? I want to order something online but I don’t know what to write for the address section. Yea, I want to buy a gift card for Elizabeth; it’s her birthday. Ok, thanks so much."""
li_comp_sent12 = """Hi Brady, could you pay me back for the money I lent you last week? Yea, wire transfer is fine. If you don’t have the money, it’s fine. Just let me know when you can pay me back."""
li_comp_sent13 = """ Hey Brandon, just checking up on you. I heard you move houses; where’s your new address? That’s great! I also heard that you lost your credit card, did you find it yet?"""
li_comp_sent14 = """Hello, this is your taxi driver. Just to confirm, is your address 29th street house 9012? Is this incorrect? If so, please give me your correct address. I will be there soon. Thank you so much."""

inp_sci_li = [li_comp_sent1, li_comp_sent2, li_comp_sent3, li_comp_sent4, li_comp_sent5, li_comp_sent6, li_comp_sent7]
inp_sci_li2 = [li_comp_sent8, li_comp_sent9, li_comp_sent10, li_comp_sent11, li_comp_sent12, li_comp_sent13, li_comp_sent14]

### 보이스피싱인지 판단하는 원본 스크립트
scripts = """IRS Phone Scam – call transcript Page 1 of 17
        This is a transcript of a call between a Pindrop Security employee, identifying himself as
        R. Williams, and a phone fraudster identifying himself as S. Parker. Pindrop identified
        one of the phone numbers used by a ring of fraudsters alleging to be working with the
        IRS to collect taxes. Pindrop called the number and purported to be R. Williams, a
        victim.
        S. Parker: Thank you for calling. How can I help you today?
        R. Williams: Hi. I got a call from this number. A recording saying to call back.
        Something about my tax paperwork.
        S. Parker: What's your name?
        R. Williams: My name is Richard Williams.
        S. Parker: Richard Williams?
        R. Williams: Yes.
        S. Parker: We don't have anybody Richard Williams.
        R. Williams: I'm sorry, who is this?
        S. Parker: Who are you?
        R. Williams: I got a call. Something about my tax paperwork having a problem.
        S. Parker: Which number you received the call? Tell me the number on which
        you received the call.
        R. Williams: 404-295-9315.
        S. Parker: You received the phone call today?
        R. Williams: Yeah, today or last night. I'm trying to find out who I'm dialing. I
        couldn't make out the message.
        S. Parker: Federal Investigation Department.
        R. Williams: I'm sorry?
        S. Parker: Federal Investigation Department.
        R. Williams: Federal Investigation Department?
        IRS Phone Scam – call transcript Page 2 of 17
        S. Parker: Yes.
        R. Williams: Okay. I mean, is it part of the IRS or something? State?
        S. Parker: This is a legal Department working for Internal Revenue Service.
        R. Williams: Okay. What do you mean by legal? You got me a little nervous
        here. I'm trying to find out what's going on.
        S. Parker: What is your native country, sir
        R. Williams: I didn't hear you. What was that?
        S. Parker: Which is your native country? Which country belong to originally?
        R. Williams: Which country I belong to originally? I'm an American.
        S. Parker: You have born and brought up an American?
        R. Williams: I have what?
        S. Parker: You have born and brought up in American?
        R. Williams: Yeah. American.
        S. Parker: All right. We don't call Americans, sir.
        R. Williams: You don't call Americans? Okay.
        S. Parker: Yes.
        R. Williams: Then I'm really confused. Who do you call?
        S. Parker: The people who evade taxes.
        R. Williams: The people who evade tax boards?
        S. Parker: Taxes. Taxes. Income taxes.
        R. Williams: Oh, taxes . Yeah, well, so that's the weird thing. They said there
        was something with my taxes. I couldn't make out the ....That's the
        message. Something about my taxes, but it was really broken up. I
        think I got a bad reception here again.
        S. Parker: All right. Did you pay your taxes. Have you filed your taxed.
        IRS Phone Scam – call transcript Page 3 of 17
        R. Williams: I filed my taxes.
        S. Parker: You filed your taxes or your CPA files it?
        R. Williams: I have an accountant that files it for me.
        S. Parker: Why then he is doing many mistakes in the filing of taxes?
        R. Williams: Why is he what? I'm sorry. I can't hear you too well.
        S. Parker: Sir, are you doing some overseas transactions?
        R. Williams: I have some investments.
        S. Parker: Because according to the rules and regulations, when there are
        overseas transactions under your name, you need to pay the full
        portion of the transaction fees to the IRS Department, which you
        never did. They have investigated each and every thing, and they
        have filed a lawsuit complaint against your name.
        R. Williams: Are you sure about that? I mean, these are just like miscellaneous
        investments, and like some stocks and company stuff. It was
        nothing really big. I think my accountant has taken care of all that.
        S. Parker: Still, the transactions are being made for the overseas, and the tax
        are being pending under you name for the overseas transaction. I'm
        not talking about the income tax. You pay you pay your income tax
        perfectly. It has been filed perfectly every year. Nothing is problem
        in that. But I'm talking about the overseas transactions which are
        made from your end. For that you need to pay the full portion of the
        tax of amount to the IRS, which you never did. Which you never
        showed at the time of your tax filing as well.
        R. Williams: Okay. So, I mean, how much are we talking about?
        S. Parker: It's $5,868. The pending amount under your name.
        R. Williams: Oh my god. Okay. I mean, are we sure this is for me?
        S. Parker: This is including the penalty amount, the late payment charges, as
        well as all of the court resolution fees, because the case has
        already been filed under you name.
        R. Williams: Okay, what do you mean by a case?
        IRS Phone Scam – call transcript Page 4 of 17
        S. Parker: In a few days, the arrest warrant will also be issued under your
        name.
        R. Williams: An a-what?
        S. Parker: Under your name. The arrest warrant will also be issued under your
        name.
        R. Williams: An IRS warrant?
        S. Parker: Arrest warrant, sir. Arrest.
        R. Williams: Arrest warrant? I got an arrest warrant under my name?
        S. Parker: Yes. You will be getting it soon. Because you are running out of the
        situation right now. You have evaded the tax, and you are under
        their criminal records right now.
        R. Williams: But I...Okay, so, I didn't do anything. I mean, I have an accountant
        that takes care of this for me. So, is there something we can talk to
        and just get this cleared up. I mean, I'm sure it's just a minor issue.
        S. Parker: Sir. Are you there?
        R. Williams: I mean, I'm sorry. I didn't hear you.
        S. Parker: Hello?
        R. Williams: Yes.
        S. Parker: You need to clear out the taxes, the tax amount which is pending
        under your name.
        R. Williams: Okay, I'm sorry. I didn't even catch your name.
        S. Parker: Steve Parker.
        R. Williams: Okay, Mr. Parker. So, I mean, here's the thing. I don't have that
        type of money all in one shot. Is there something that, I don't know.
        I mean, in the past if I had any taxes, I would just make a deposit...
        S. Parker: Sir? We are not...Sir? We are not bill collectors. I'm not asking you
        for any money from you. You are not paying this amount to me. So,
        if you don't have the funds, then don't talk to me. Because I am
        telling you to submit this payment to the IRS Department itself.
        Because we are doing an offer in compromise with them regarding 
        IRS Phone Scam – call transcript Page 5 of 17
        your case. Because your case is being forwarded to your local
        county sheriff department. You will be receiving an arrest warrant
        as soon as possible. I think it will be tomorrow itself.
        R. Williams: Okay, so, how? I don't want to go to jail. So what do I need to do to
        take care of this? Who do I call?
        S. Parker: You need to pay this amount: $5,868.
        R. Williams: I need to pay the full amount?
        S. Parker: Yes, the full amount. This is a compromise amount which you need
        to submit.
        R. Williams: What is compromise amount?
        S. Parker: Offer in compromise.
        R. Williams: Okay, is there a case number?
        S. Parker: 11/100/3636.
        R. Williams: 11/100/3636.
        S. Parker: 3636, yes.
        R. Williams: Okay. And who do I make the payment to?
        S. Parker: You will be making this payment....
        R. Williams: Can you guys take a credit card?
        S. Parker: No, we don't accept any of your credit cards, any of your debit
        cards, any of your bank account information. Because you need to
        pay this amount by the tax pay vouchers, which are available to any
        of the government stores, federal stores nearby you. I will guide
        you about the stores, but you need to have this amount cash with
        you.
        R. Williams: Okay. And what do I do? There's a tax voucher or something at one
        of my local stores?
        S. Parker: Not local stores, sir. Government stores.
        R. Williams: Okay. What government stores would that be?
        IRS Phone Scam – call transcript Page 6 of 17
        S. Parker: Home Depot, Food Lion.
        R. Williams: Okay. And what do I ask them for? You say you don't take credit
        card or Visa.
        S. Parker: You need to ask them about the tax pay vouchers . There will be
        tax pay vouchers available to that store. And you just need to take
        the cash over there, upload the funds from that tax pay voucher so
        that we can submit that voucher to the IRS Department for the offer
        in compromise. Once the payment will be submitted you will be
        provided with the receipt copy as well as the clearance letter from
        the IRS Department as well.
        R. Williams: Okay, so I just go over there, put the money into a tax pay voucher
        and mail it in?
        S. Parker: Yes.
        R. Williams: Okay. And where do I mail it to?
        S. Parker: I will put all the information. I will write all the information. You don't
        have to worry about that. First of all, you need to arrange the funds,
        $5,868.
        R. Williams: Okay. That's fine. I'm going to get that right now. Then do I mail it
        in, or do I call you back. What do you need me to do, then. What
        am I supposed to do?
        S. Parker: Arrange the funds. Call me back, so I guide you to the nearest store
        to you. What is your zip code, by the way.
        R. Williams: The zip code I'm in at the moment is 30303.
        S. Parker: Hold on. What's your zip code you told me?
        R. Williams: I'm actually traveling. This is great.
        S. Parker: What's the zip code you told me? What's the zip code, sir? 30....
        R. Williams: 303. I'm in Atlanta right now.
        S. Parker: Sir, please repeat the zip code one more time.
        R. Williams: I'm in 30303. I'm in downtown Atlanta.
        IRS Phone Scam – call transcript Page 7 of 17
        S. Parker: Okay, okay. Let me give you a location. All right. You have a Home
        Depot. 650 Ponce de Leon, Atlanta.
        R. Williams: Yeah, I'm sure I can...I think I saw one driving in here. Okay? So, I
        go there? And then, that's fine.
        S. Parker: First you need to have cash, sir. First you need to have cash. Do
        you have this amount cash with you?
        R. Williams: Oh my goodness. I didn't even realize what time it was. I don't have
        all the cash with me.
        S. Parker: Hello?
        R. Williams: I don't have all the cash with me and the bank's already closed. I'm
        going to have to do it first thing in the morning.
        S. Parker: How much amount do you have? How much amount do you have
        right now? We need to cancel your arrest....
        R. Williams: I think like $2600.
        S. Parker: $2600?
        R. Williams: Yeah, that's all I have with me at the moment.
        S. Parker: You need to submit $2400 for the warrant cancellation fees today.
        R. Williams: Okay. Okay. So, you'll accept that for now, and tomorrow morning I
        can take care of the rest?
        S. Parker: Yes.
        R. Williams: Okay. Okay. When I call back, is this your direct number, or...
        S. Parker: Yes, this is my direct line number.
        R. Williams: Okay, Mr. Parker. Let me go run real quick, get that tax voucher.
        S. Parker: You can be on hold with me, sir. You can be on hold so that we will
        have the evidence for this recording as well that you tried your best
        in order to arrange the half of the payment. Because you are not
        paying this amount in one shot, right?
        IRS Phone Scam – call transcript Page 8 of 17
        R. Williams: Yeah, but I mean, I need to go wash up real quick. If you can give
        me 20 minutes. It's not exactly walking distance. I need to go drive
        there. So can I call you back?
        S. Parker: Sir, I need you to keep your phone on loudspeaker so that I am
        preparing the documentation for the cancellation of the arrest
        warrant. If I have any queries regarding, I need to ask you. So it's
        better that you keep your cell phone with you on a loudspeaker, so
        that I can talk to you.
        R. Williams: Okay. But the phone might die, getting into the elevator and stuff.
        S. Parker: You don't have a car charger with you?
        R. Williams: Yes, yes. I just don't have reception when I go down to the lobby in
        the hotel. I told you that I'm traveling. I'm not in the city.
        S. Parker: Okay, not a problem. So once you reach the parking lot of the
        Home Depot, but are you sure you have the cash of $2400 with
        you?
        R. Williams: Yeah. I have $2600 with me. You told me you'd take $2400 just to
        make sure I don't get arrested tomorrow. Right?
        S. Parker: Yes, for the cancellation.
        R. Williams: Okay, then. Let me go get that real quick, and I will call you right
        back. It will be 10 - 15 minutes, so I can go get whatever voucher it
        is. Okay?
        S. Parker: You need to call me once you reach the parking lot, first of all, so
        that I will guide you which tax pay voucher you need to obtain.
        R. Williams: Okay, okay. Okay.
        S. Parker: Okay? So, once...
        R. Williams: You know what? There's a Walmart across the street. Is there a tax
        voucher there?
        S. Parker: No, Walmart people. That's not a government store, sir. I don't
        know whether they keep it or not, but I'm not sure about that.
        R. Williams: Okay, okay. Okay. Give me two minutes. Let me go put on some
        clothes real quick, and I'll call you when I go downstairs.
        IRS Phone Scam – call transcript Page 9 of 17
        S. Parker: All right?
        R. Williams: Okay, Mr. Parker. Thank you.
        S. Parker: Not a problem.
        R. Williams: Bye.
        S. Parker: Bye bye.
        (Phone ringing):
        S. Parker: This is Steve Parker.
        R. Williams: Mr. Parker.
        S. Parker: Yes.
        R. Williams: I was talking to Brian and I got disconnected.
        S. Parker: Okay, hold on. Let me transfer this line to Mr. Brian, okay?
        R. Williams: Okay.
        B. Jackson: Accounting Department. This is Brian.
        R. Williams: Hi, Brian. This is Mr. Williams calling you back again, sir.
        B. Jackson: Yes, sir.
        R. Williams: Sorry about that.
        B. Jackson: No problem.
        R. Williams: I'm actually crossing the street to the Walmart now.
        B. Jackson: Okay, once you reach the parking lot, before you walk in, just let
        me know.
        R. Williams: There's no parking lot here. It's literally across the street from
        where I'm staying at. It's a lot closer.
        B. Jackson: Okay, before you walk in, just let me know.
        R. Williams: Okay. I'm right outside the entrance.
        IRS Phone Scam – call transcript Page 10 of 17
        B. Jackson: Okay, now. You have to get...The restitution department where the
        payment has to be submitted. I will guide you with the procedure of
        how it has to be done. They have a PayPal account, and your
        check...If you write a check, your check cannot reach in one or two
        hours to Washington, D.C., to submit the payment.
        R. Williams: Right.
        B. Jackson: Correct?
        R. Williams: That's why you said I needed to get cash. Right.
        B. Jackson: So you have to wire that money onto the restitution's PayPal
        account. Okay? And you need to walk up to the money center.
        R. Williams: Uh huh.
        B. Jackson: In the money center, there will be different racks where you can see
        several cards available over there.
        R. Williams: Okay.
        B. Jackson: Yeah, can you reach the money center?
        R. Williams: I'm trying to get to it. This one is weird. This Walmart is actually
        pretty small.
        B. Jackson: No problem. Just ask anybody where is the money center. Once
        you see all the cards, let me know.
        R. Williams: I don't see a money center here.
        B. Jackson: Where you can see all those cards hanging around.
        R. Williams: All the cards hanging around. I'm so nervous here right now. Oh
        dear, oh dear.
        B. Jackson: Don't worry, sir. You will find them. Just ask anybody.
        R. Williams: I'm trying to find someone here. It's 5:02 and I can't find anyone.
        Oh dear. I see some cards here. Okay. I see some cards. What
        do you want me to with the cards?
        B. Jackson: Okay. Which all you can see? Can you just give me one or two
        names?
        IRS Phone Scam – call transcript Page 11 of 17
        R. Williams: Yes. I see in-store credits, gift cards, Subway, Starbuck's.
        B. Jackson: Okay. Can you see...We need it for a PayPal account, which is
        known as Green-dot-money pack.
        R. Williams: A Green-dot-money pack.
        B. Jackson: You tell them you need Green-dot-money pack for PayPal account.
        R. Williams: Money pack? Yeah, they got it behind there.
        B. Jackson: Okay, yeah. Just take three of them in your hand.
        R. Williams: He's got it behind the register here. So, what do you want to do?
        Ask him for three?
        B. Jackson: Okay, just tell them on one money pack you can only upload
        $1,000. So tell them you need three of them: two for $1,000 and
        one for $400.
        R. Williams: Okay. I'm waiting in line. So, bear with me, sir.
        B. Jackson: No problem. If there is a limit, one money...It is a green color
        coupon. Can you see that? It's a green color voucher.
        R. Williams: I'm sorry?
        B. Jackson: It is a green color voucher. Can you see the money pack, green
        color?
        R. Williams: Yes, it was green color. He showed me.
        B. Jackson: Okay, great. Just tell that you need three of them: two for $1,000,
        one for $400.
        R. Williams: Okay. Two for $1,000.
        B. Jackson: Okay? And once you have uploaded...two for $1,000 and one for
        $400. Once you upload the funds, once you made the payments,
        they will give you a receipt copy. Once you get the receipt copy,
        just let me know. I will hold on.
        R. Williams: Okay.
        B. Jackson: Take the cards, take the receipt copy, and say Hello, and upload
        the funds.
        IRS Phone Scam – call transcript Page 12 of 17
        R. Williams: I'm waiting. I understand. I'm waiting for the person in front of me.
        B. Jackson: No problem, sir. No problem. I will hold on. No problem.
        R. Williams: How much are the green cards?
        Clerk: About $5.
        R. Williams: Okay. Can I get three? Two for $1,000 and one for $500? Yeah,
        two for $1,000, one for $400. Hello?
        B. Jackson: Yes, sir. I'm here.
        R. Williams: Okay, so I'm walking back now.
        B. Jackson: Okay, you got the three cards and the receipt?
        R. Williams: I'm sorry. I can't hear you. You're breaking up. Hello?
        B. Jackson: Okay, you got the cards and the receipt copy?
        R. Williams: Yes.
        B. Jackson: Okay. Once you reach your room, just let me know. I will tell you
        how to use it.
        R. Williams: Okay. But I'm telling you once I get to the lobby, I don't have any
        reception.
        B. Jackson: I understand.
        R. Williams: Oh my goodness. I'm going to need a big drink after this. Oh dear,
        oh dear. If it's not one thing, it's another. Okay. All right.
        (Phone ringing):
        S. Parker: Thank you for calling. This is Steve Parker. How can I help you?
        R. Williams: Mr. Parker. Okay, sir.
        S. Parker: Yes.
        R. Williams: This is Mr. Williams again. I need to talk to Brian?
        S. Parker: Okay, hold on. Let me transfer this line.
        IRS Phone Scam – call transcript Page 13 of 17
        B. Jackson: Brian Jackson. Accounting Head.
        R. Williams: Okay, Mr. Jackson. This is Mr. Williams again. Sorry about that.
        B. Jackson: Yes, sir. Do you have the packs right here? No problem.
        R. Williams: Yeah, I got all the stuff with me. The three cards. All right.
        B. Jackson: Yeah. Flip the card one by one. You will see a silver color scratchoff part on the backside.
        R. Williams: Yeah, I see it here.
        B. Jackson: Yeah, scratch it off carefully, and make sure you don't lose any
        numbers. And help me out with the numbers one by one which you
        can see below that.
        R. Williams: And you want to do what?
        B. Jackson: Once you scratch it off, you will see some numbers, which is used
        to wire the money to the PayPal account of the restitution
        department. And do you have the receipt copy with you?
        R. Williams: Yeah, I do.
        B. Jackson: First of all, just help me out with the store number on the receipt
        copy. There will be ST written on that.
        R. Williams: All right. Store number 3775.
        B. Jackson: Okay. And the store phone number?
        R. Williams: All right. Looks like it's 404-352-5252.
        B. Jackson: 404-352-5252. Okay. Yeah. Now, at the backside, scratch it off the
        first card and help me out with the number.
        R. Williams: Okay.
        B. Jackson: Once you scratch it off, the number you can see just after. Help me
        out with that.
        R. Williams: Okay, what do you want me to do? Do you want me to just give
        you this number?
        B. Jackson: Yeah, you have to give me those numbers.
        IRS Phone Scam – call transcript Page 14 of 17
        R. Williams: Oh, no. I don't know about that. Aren't I supposed to go to a
        PayPal.com website or something like that for this? How do I
        know...
        B. Jackson: That is a receipt copy for your information, sir, which has been
        given to you. And we will just verify that. We will just mention that
        you bought it from the store with phone number 3775. And if there
        are some problems to verify that, we will call them and verify the
        transaction.
        R. Williams: I mean, can't I just do this. I'd feel a little bit more comfortable if I
        can do this online so print out the receipt right away in case the
        cops show up or something. I can at least show them, hey, look I
        took care of something.
        B. Jackson: Yes, you are to bring the receipt copy with you. Sir, you are keeping
        that receipt copy with you for your conformation that, yes, you are
        the owner of those money packs and you made those payments.
        This is what the reason....
        R. Williams: Yeah, but you know, if the cop comes here, he's not going to care
        about I got some receipt here. He's going to be like, "I got a
        warrant here for your arrest unless you have something, you
        know."
        B. Jackson: Yes, sir. Once you pay those off. Once we withdraw the warrant,
        you will get an out-of -court restitution certificate. That will be...Yes.
        R. Williams: Okay, and how do I get that?
        B. Jackson: If you have a fax machine, once you make the payments, once we
        withdraw the charges under your name and put your case on hold,
        then they will issue an out-of-court restitution certificate. If you have
        a fax machine around you, we can fax it to you. Or if you don't have
        access to a fax machine, as you told me you're in a hotel, so we
        can email it you.
        R. Williams: Yeah, the lobby might have a little office place or something. But, I
        don't know.
        B. Jackson: No problem. But do you have access to your email? We can scan
        that copy and send it to your email as well. No problem with that.
        R. Williams: Man, all right. I got nothing here to scratch it with. It's just crazy.
        Like I said, if I could go through the IRS website and pay this
        online.
        IRS Phone Scam – call transcript Page 15 of 17
        B. Jackson: I understand your concern, sir. I understand. But the IRS would not
        give you certain information. That's the whole point. As the reason
        as IRS penalty accepts check. But as it was a short notice that we
        need to drop these charges under your name, we need to directly
        submit this payment to the restitution department, and make an
        assurance that, yes, your case has been put on hold and you get
        your out-of-court restitution certificate. So that we can make your
        records clean and make is as clear as it was in the past.
        R. Williams: Yeah, but all I want to do...
        B. Jackson: But sir, if you have a problem about that, I will connect this line to
        Mr. Stephen Parker, and you can go and talk to him that you don't
        want to make the payments, and he can do whatever he needs to
        do. I'm just an accounting department guy, sir. My job is to just
        submit the payments to the restitution department and guide you
        how you make the payments. If you have any problems with
        payments...I'll connect your line to Stephen Parker.
        R. Williams: So, if I give you this number...I mean the thing is, how do I get
        proof?
        B. Jackson: I promise you one thing once you help me out with numbers, I'm not
        going to dominate this phone call. I will be on hold with you on the
        same phone call I will forward the information to the restitution
        department on their PayPal account. And unless and until you get
        your out-of-court restitution certificate, I am not going to disconnect
        the call from you.
        R. Williams: Yeah, you know, no. I need to turn around. I'll feel a little more
        comfortable if I can get some proof right away. You can't fax it to
        me or something. I get nervous about that. I need to think about
        this. I'll call you back if anything. No, no, no. I have to talk to my
        accountant.
        B. Jackson: Sir, you have to tell this to Stephen Parker, sir. You have to tell this
        to Stephen Parker that you're not making the payments, and he can
        issue the arrest warrant or do whatever he wants to. Just hold the
        line. I'll connect the line to Stephen Parker, okay? Tell him that
        you're not making the payments, and we will go from there. Okay?
        Just hold on for a second.
        R. Williams: Okay. Okay.
        S. Parker: Steve Parker.
        IRS Phone Scam – call transcript Page 16 of 17
        R. Williams: Yeah, Mr. Parker. I was speaking with Brian. But I can't just take
        this without some type of proof on my side protecting the money
        goes out. Cop comes here, knocks on my door. I can't.
        S. Parker: I will explain everything. Okay, sir?
        You are making this payment to the restitution department directly,
        sir. As it was a short notice, and this was a final notification phone
        call as I told you before, as well. This was a final notification phone
        call. I don't want to pay this amount, sir. I told you first, as well, we
        are not concerned about the amount. We are concerned about you.
        Because, as we looked into your past records, you have very clean
        record. You are a good citizen.
        R. Williams: Well, I always pay my stuff on time. That's what I pay an
        accountant for.
        S. Parker: This is the reason that we have given you this final notification
        phone call. If you would have been a criminal, then we would have
        never given you a phone call, and we have directly served you with
        an arrest warrant, sir. The reason behind the phone....
        R. Williams: Okay, so let's me ask you a question. Can I go online somewhere,
        to turn around at least see this information?
        S. Parker: What information you want, sir? I already told you you are running
        out of your time. I give you a chance in order to resolve this issue. I
        will provide you time until tomorrow. But you need to cancel your
        arrest warrant today, sir. If you will not pay this, I will also not be
        able to help you out in this case. And you're not making your
        payment.
        R. Williams: I understand you're trying to help.
        S. Parker: Sir, you are not making your full payment today. Then also I told
        Mr. Brian Jackson that please issue him a clearance letter from the
        courthouse, so that if he is in any sort of problem, he can have
        proof with him that, yes, he has made a partial payment and has
        the records with him. This was the reason we are trying to help you.
        R. Williams: But my problem is I just need something so I turn around when a
        cop come here, I can say, "Look. I made a payment. Here's the
        case number. And here's the receipt." I just don't know. I don't
        know. 
        IRS Phone Scam – call transcript Page 17 of 17
        S. Parker: Sir, I told you, sir. I told you. You will get each and everything once
        they received payment. Before making the payment, how can we
        provide the receipt copy and the clearance letter? How can we get
        you that. Once the payments are submitted to them, within 5 to 10
        minutes, your clearance letter will be coming from there, and we will
        mail it to you on the same phone call. We will not disconnect this
        line until the time you will not get your clearance letter. Because
        you are..
        R. Williams: I got to talk to my accountant, Mr. Parker. I appreciate it. I
        understand what you're saying. But I got to talk to my accountant
        and get this straightened out. I will call you in the morning, first
        thing in the morning. I will call him right now.
        S. Parker: Not a problem, sir. I think you are giving me a flat refusal. Now I
        also want...(unitelligible)
        R. Williams: I didn't understand a word you said. You're breaking up really bad.
        S. Parker: I'm telling you, sir. We are now taking this matter in legal ways,
        because I think you are running out of the situation, and I will
        forward this file to your local county sheriff department right now.
        And make sure if you receive any phone call from you local county
        sheriff department, you receive their phone call. Because they will
        come to arrest you, sir.
        R. Williams: I want to talk to my accountant. I understand, Mr. Parker. I've just
        got to take this quick chance here. But you know. I need to make
        everything okay.
        S. Parker: Not a problem. You are making a flat refusal on this recorded line,
        sir. It will go against you inside the courthouse, as well. Not a
        problem. Take care.
        R. Williams: Okay. I will verify. Thank you.
        S. Parker: Thank you."""




# 키워드 비교, 문장비교를 통한 보이스피싱 분석 결과
if __name__ == "__main__":
    vp = VP()
    result_list_A = []
    result_list_B = []
    for i in inp_sci_li:
        re_score = vp.simli(i)
        result_list_A.append(round(re_score))
        re_keywordComp = vp.vali_voice_phi(scripts, i)
        result_list_B.append(round(re_keywordComp))
    print("보이스피싱 맥락적 유사성 분석결과 A  :", result_list_A)
    print("보이스피싱 키워드 비교 분석결과 B :", result_list_B)
    # 해석은 각 숫자값이 높을 수록 보이스피싱일 가능성이 높음
    # AB 는 보이스피싱 문장을 분석한 결과
    # 보이스피싱 맥락적 유사성 분석결과 A  : [53, 52, 47, 52, 69, 53, 43]
    # 보이스피싱 키워드 비교 분석결과 B : [80, 47, 59, 68, 57, 54, 45]


    result_list_C = []
    result_list_D = []
    for k in inp_sci_li2:
        re_score_ = vp.simli(k)
        result_list_C.append(round(re_score_))
        re_keywordComp_ = vp.vali_voice_phi(scripts, i)
        result_list_D.append(round(re_keywordComp_))
    print("보이스피싱 맥락적 유사성 분석결과 C :", result_list_C)
    print("보이스피싱 키워드 비교 분석결과 D :", result_list_D)
    # 해석은 각 숫자값이 높을 수록 보이스피싱일 가능성이 높음
    # CD 는 보이스피싱이 아닌 문장을 분석한 결과
    # 보이스피싱 맥락적 유사성 분석결과 C : [56, 81, 42, 48, 51, 46, 49]
    # 보이스피싱 키워드 비교 분석결과 D : [45, 45, 45, 45, 45, 45, 45]
    






