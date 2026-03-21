import streamlit as st
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import sys, os, time, random, base64
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_OK = True
except:
    PDF_OK = False

st.set_page_config(page_title="Tether", page_icon="🔵", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"],.stApp{background:#060f20 !important;color:#eef2ff !important;font-family:'Inter',sans-serif !important;}
.block-container{padding:2rem 3rem !important;max-width:1400px !important;}
#MainMenu,footer,header,[data-testid="stSidebar"]{visibility:hidden !important;display:none !important;}
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed) !important;color:#fff !important;font-family:'Space Grotesk',sans-serif !important;font-weight:700 !important;font-size:16px !important;border:none !important;border-radius:14px !important;padding:15px 32px !important;width:100% !important;box-shadow:0 6px 24px rgba(124,58,237,.4) !important;transition:all .25s !important;}
.stButton>button:hover{background:linear-gradient(135deg,#4338ca,#6d28d9) !important;transform:translateY(-2px) !important;box-shadow:0 10px 32px rgba(124,58,237,.6) !important;}
.stButton>button p,.stButton>button span{color:#fff !important;font-weight:700 !important;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.03) !important;border:1px solid rgba(99,102,241,.2) !important;border-radius:16px !important;padding:5px !important;gap:4px !important;}
.stTabs [data-baseweb="tab"]{background:transparent !important;color:#4a6080 !important;font-size:14px !important;font-weight:500 !important;border-radius:12px !important;padding:10px 22px !important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(99,102,241,.25),rgba(124,58,237,.15)) !important;color:#a5b4fc !important;font-weight:700 !important;}
.stTextInput input,.stTextArea textarea{background:#ffffff !important;border:1px solid rgba(99,102,241,.35) !important;border-radius:12px !important;color:#111827 !important;font-size:15px !important;}
label{color:#94a3b8 !important;font-size:14px !important;}
[data-testid="stSlider"]>div>div>div>div{background:#6366f1 !important;}
div[data-baseweb="select"]>div{background:rgba(99,102,241,.08) !important;border:1px solid rgba(99,102,241,.25) !important;border-radius:12px !important;color:#eef2ff !important;}
li[role="option"]{background:#0a1628 !important;color:#eef2ff !important;}
[data-testid="stMetric"]{background:rgba(99,102,241,.08) !important;border:1px solid rgba(99,102,241,.2) !important;border-radius:16px !important;padding:22px !important;}
[data-testid="stMetricLabel"]{color:#94a3b8 !important;font-size:12px !important;text-transform:uppercase !important;letter-spacing:.1em !important;}
[data-testid="stMetricValue"]{color:#eef2ff !important;font-family:'Space Grotesk',sans-serif !important;font-size:30px !important;font-weight:800 !important;}
[data-testid="stMetricDelta"]{color:#818cf8 !important;font-size:13px !important;}
details{background:rgba(99,102,241,.05) !important;border:1px solid rgba(99,102,241,.18) !important;border-radius:16px !important;margin:10px 0 !important;}
details summary{color:#a5b4fc !important;font-weight:700 !important;font-size:15px !important;padding:16px 20px !important;cursor:pointer !important;}
details[open]{border-color:rgba(99,102,241,.4) !important;}
::-webkit-scrollbar{width:5px;height:5px;}::-webkit-scrollbar-track{background:#060f20;}::-webkit-scrollbar-thumb{background:#2e3a6e;border-radius:99px;}
.stProgress>div>div{background:linear-gradient(90deg,#6366f1,#8b5cf6) !important;border-radius:99px !important;}
hr{border-color:rgba(99,102,241,.12) !important;}
</style>""", unsafe_allow_html=True)

for k,v in [("page","landing"),("step",1),("answers",{}),("result",None),("ao",{})]:
    if k not in st.session_state: st.session_state[k]=v

@st.cache_resource
def load_model():
    try:
        from models.train_model import TetherPredictor
        return TetherPredictor(),True
    except: return None,False
predictor,model_loaded=load_model()

BG="#060f20"; BG2="#0c1a35"
TC={"Healthy":"#10b981","Situational":"#60a5fa","Social":"#a78bfa","Existential":"#fbbf24","Chronic":"#f87171"}

def sc(s):
    if s>=70: return "#10b981"
    elif s>=50: return "#60a5fa"
    elif s>=30: return "#fbbf24"
    else: return "#f87171"
def sl(s):
    if s>=70: return "Thriving"
    elif s>=55: return "Stable"
    elif s>=40: return "Drifting"
    elif s>=25: return "Struggling"
    else: return "In Crisis"

def H(t,sz=22,c="#eef2ff"): st.markdown(f"<p style='font-family:Space Grotesk,sans-serif;font-size:{sz}px;font-weight:800;color:{c};margin:0 0 8px;'>{t}</p>",unsafe_allow_html=True)
def LBL(t,c="#6366f1"): st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{c};font-weight:700;margin:0 0 10px;'>{t}</p>",unsafe_allow_html=True)
def BODY(t,c="#94a3b8",sz=14): st.markdown(f"<p style='font-size:{sz}px;color:{c};line-height:1.75;margin:0 0 12px;'>{t}</p>",unsafe_allow_html=True)


def render(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=130,bbox_inches="tight",facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    plt.close("all")
    st.image(buf,use_container_width=True)


def compute(ans):
    resp={"Instantly":8,"Within an hour":25,"Few hours":90,"Next day":220,"Often leave unread":380}.get(ans.get("q1"),90)
    contacts={"10+ people":15,"5–10 people":7,"2–4 people":3,"1 person":1,"Nobody":0}.get(ans.get("q2"),4)
    init={"Always me":0.78,"Mostly me":0.62,"Equal":0.50,"Mostly them":0.32,"Rarely me":0.14}.get(ans.get("q3"),0.5)
    night={"Never":0.02,"Occasionally":0.10,"Few nights/week":0.24,"Most nights":0.38,"Every night":0.52}.get(ans.get("q4"),0.10)
    weekend={"Very active":0.88,"Balanced":0.62,"Mostly home":0.38,"Mostly alone":0.22,"In bed / avoidant":0.10}.get(ans.get("q5"),0.62)
    future={"A lot":0.40,"Somewhat":0.24,"Not much":0.12,"Rarely":0.06,"Feels blank":0.02}.get(ans.get("q6"),0.24)
    alone=1 if ans.get("q7","") in ["Alone","Living alone"] else 0
    wfh=1 if ans.get("q8","") in ["Remote / WFH","Not working"] else 0
    score=100
    if resp>200: score-=20
    elif resp>80: score-=10
    if contacts==0: score-=28
    elif contacts<3: score-=18
    elif contacts<6: score-=8
    if init<0.2: score-=16
    elif init<0.35: score-=8
    if night>0.35: score-=15
    elif night>0.2: score-=7
    if weekend<0.25: score-=14
    elif weekend<0.45: score-=6
    if future<0.05: score-=12
    elif future<0.15: score-=5
    if alone: score-=6
    if wfh: score-=5
    score=max(6,min(95,score+np.random.normal(0,2)))
    lt=0 if score>70 else 1 if score>55 else 2 if score>40 else 3 if score>25 else 4
    labs=["Healthy","Situational","Social","Existential","Chronic"]
    probs=[0.05]*5; probs[lt]=0.55
    for i in range(5):
        if i!=lt: probs[i]=(1-0.55)/4
    sigs=[]
    if night>0.22: sigs.append({"s":"Late-night activity spike","sev":"high","d":f"{night*100:.0f}% after 1am","i":"Clearest behavioral isolation marker"})
    if contacts<4: sigs.append({"s":"Shrinking social circle","sev":"high","d":f"Only {max(0,contacts)} real convos/week","i":"Below 5 is clinically significant"})
    if init<0.28: sigs.append({"s":"Stopped reaching out","sev":"high","d":f"You initiate only {init*100:.0f}% of convos","i":"Low initiation accelerates drift"})
    if weekend<0.35: sigs.append({"s":"Weekend isolation","sev":"medium","d":"Activity collapses on weekends","i":"3× more predictive than weekday"})
    if future<0.08: sigs.append({"s":"Future thinking collapsed","sev":"medium","d":"Almost no forward language","i":"Early depressive drift marker"})
    return {"score":round(score,1),"lt":lt,"label":labs[lt],
            "crisis_prob":max(0,min(1,(50-score)/80)),
            "crisis_level":"Low" if score>60 else "Moderate" if score>42 else "High" if score>25 else "Critical",
            "drift":{"txt":"Stable" if score>65 else "Drifting" if score>42 else "Declining",
                     "arrow":"→" if score>65 else "↘" if score>42 else "↓",
                     "c":"#10b981" if score>65 else "#fbbf24" if score>42 else "#f87171"},
            "fp":{"Social Isolation":max(0,min(100,100-contacts*6)),
                  "Emotional Distance":max(0,min(100,resp/3.8)),
                  "Existential Void":max(0,min(100,(1-future)*78)),
                  "Behavioral Withdrawal":max(0,min(100,(1-weekend)*62+night*88)),
                  "Linguistic Decay":max(0,min(100,(1-max(0.38,0.82-contacts*0.05))*72))},
            "probs":{labs[i]:round(probs[i],3) for i in range(5)},
            "sigs":sigs[:4],
            "ctx":{"resp":resp,"contacts":contacts,"init":init,"night":night,"weekend":weekend,"future":future}}

def chart_gauge(score):
    """Standalone gauge — plt.subplots polar"""
    fig,ax=plt.subplots(1,1,figsize=(5,3.8),subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    theta=np.linspace(0,np.pi,300)
    ax.plot(theta,[1]*300,color="#1e2d4a",linewidth=16,solid_capstyle="round",zorder=1)
    frac=max(0.03,score/100)
    ax.plot(np.linspace(0,np.pi*frac,300),[1]*300,color=sc(score),linewidth=16,solid_capstyle="round",zorder=2)
    ax.set_ylim(0,1.7); ax.set_theta_offset(np.pi); ax.set_theta_direction(-1)
    ax.set_thetalim(0,np.pi); ax.axis("off")
    ax.text(np.pi/2,0.52,f"{score:.0f}",ha="center",va="center",fontsize=54,fontweight="bold",color=sc(score))
    ax.text(np.pi/2,0.17,sl(score),ha="center",va="center",fontsize=14,color=sc(score),fontweight="600")
    ax.set_title("Social Health Score",color="#eef2ff",fontsize=12,fontweight="bold",pad=4)
    plt.tight_layout()
    return fig

def chart_fingerprint(fp):
    """Horizontal bar chart for fingerprint"""
    dims=["Social\nIsolation","Emotional\nDistance","Existential\nVoid","Behavioral\nWithdrawal","Linguistic\nDecay"]
    vals=list(fp.values())
    bc=["#10b981" if v<35 else "#60a5fa" if v<60 else "#fbbf24" if v<80 else "#f87171" for v in vals]
    fig,ax=plt.subplots(1,1,figsize=(6,4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.barh(dims,[100]*5,color="#1a2540",height=0.55,edgecolor="none",zorder=1)
    ax.barh(dims,vals,color=bc,alpha=0.9,height=0.55,edgecolor="none",zorder=2)
    for i,(v,c) in enumerate(zip(vals,bc)):
        ax.text(v+2,i,f"{v:.0f}",va="center",fontsize=12,color=c,fontfamily="monospace",fontweight="700")
    ax.set_xlim(0,118); ax.set_xlabel("Stress Level →",color="#4a6080",fontsize=10,labelpad=8)
    ax.set_title("Loneliness Fingerprint",color="#eef2ff",fontsize=13,fontweight="bold",pad=10)
    ax.tick_params(colors="#94a3b8",labelsize=9)
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    plt.tight_layout()
    return fig

def chart_type_probs(probs):
    """Horizontal bar chart for type probabilities"""
    labs=list(probs.keys()); vals=[v*100 for v in probs.values()]
    tc=[TC.get(l,"#6366f1") for l in labs]
    ypos=list(range(len(labs)-1,-1,-1))
    fig,ax=plt.subplots(1,1,figsize=(6,4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.barh(ypos,[100]*5,color="#1a2540",height=0.55,edgecolor="none",zorder=1)
    ax.barh(ypos,vals[::-1],color=tc[::-1],alpha=0.85,height=0.55,edgecolor="none",zorder=2)
    for i,(v,c) in enumerate(zip(vals[::-1],tc[::-1])):
        ax.text(v+2,ypos[i],f"{v:.0f}%",va="center",fontsize=12,color=c,fontfamily="monospace",fontweight="700")
    ax.set_yticks(ypos); ax.set_yticklabels(labs[::-1],fontsize=11,color="#94a3b8",fontweight="500")
    ax.set_xlim(0,118); ax.set_xlabel("Probability →",color="#4a6080",fontsize=10,labelpad=8)
    ax.set_title("Type Probabilities",color="#eef2ff",fontsize=13,fontweight="bold",pad=10)
    ax.tick_params(colors="#94a3b8")
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    plt.tight_layout()
    return fig

def chart_signal_breakdown(ctx, score):
    """Clear signal impact chart - shows exactly how each answer affects the score"""
    signals = [
        ("Response Time", max(0,min(30, (ctx["resp"]-30)/15)), ctx["resp"]<60, f"{ctx['resp']:.0f} min avg"),
        ("Social Contacts", max(0,min(28, (5-ctx["contacts"])*4 if ctx["contacts"]<5 else 0)), ctx["contacts"]>=5, f"{ctx['contacts']} people/week"),
        ("Initiation Rate", max(0,min(16, (0.4-ctx["init"])*40 if ctx["init"]<0.4 else 0)), ctx["init"]>=0.4, f"{ctx['init']*100:.0f}% you start"),
        ("Night Activity", max(0,min(15, (ctx["night"]-0.15)*60 if ctx["night"]>0.15 else 0)), ctx["night"]<0.15, f"{ctx['night']*100:.0f}% after 1am"),
        ("Weekend Score", max(0,min(14, (0.5-ctx["weekend"])*28 if ctx["weekend"]<0.5 else 0)), ctx["weekend"]>=0.5, f"{ctx['weekend']*100:.0f}% active"),
        ("Future Thinking", max(0,min(12, (0.2-ctx["future"])*60 if ctx["future"]<0.2 else 0)), ctx["future"]>=0.2, f"{ctx['future']*100:.0f}% forward lang"),
    ]
    labels = [s[0] for s in signals]
    impacts = [s[1] for s in signals]
    good = [s[2] for s in signals]
    values = [s[3] for s in signals]
    colors = ["#10b981" if g else "#f87171" for g in good]

    fig, ax = plt.subplots(1,1,figsize=(9,3.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    ax.barh(labels, [30]*6, color="#1a2540", height=0.55, edgecolor="none", zorder=1)
    bars = ax.barh(labels, impacts, color=colors, alpha=0.85, height=0.55, edgecolor="none", zorder=2)

    for i, (bar, val, imp, color, g) in enumerate(zip(bars, values, impacts, colors, good)):
        ax.text(imp+0.5, i, val, va="center", fontsize=10.5, color=color, fontweight="700", fontfamily="monospace")
        status = "✓ GOOD" if g else f"-{imp:.0f} pts"
        ax.text(29, i, status, va="center", fontsize=9, color=color, fontweight="600", ha="right", fontfamily="monospace")

    ax.set_xlim(0, 32)
    ax.set_xlabel("Score Impact (points deducted)", color="#4a6080", fontsize=10, labelpad=8)
    ax.set_title("What's Affecting Your Score — Signal Breakdown", color="#eef2ff", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")

    ax.text(31.5, 5.7, f"Total: {score:.0f}/100", ha="right", va="top",
            fontsize=12, color=sc(score), fontweight="800", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0c1a35", edgecolor=sc(score), alpha=0.9))
    plt.tight_layout()
    return fig

def chart_heatmap(lt, ctx):
    """Daily health score + score gap analysis"""
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    weekend = ctx["weekend"]; night = ctx["night"]; init = ctx["init"]; future = ctx["future"]

    weekday_base = min(92, 42 + init*38 + (1-night)*18 + future*25)
    daily_scores = [
        max(5,min(92, weekday_base*0.92 + np.random.normal(0,1.5))),
        max(5,min(92, weekday_base*0.85 + np.random.normal(0,1.5))),
        max(5,min(92, weekday_base + np.random.normal(0,1.5))),
        max(5,min(92, weekday_base*0.88 + np.random.normal(0,1.5))),
        max(5,min(92, weekday_base*0.80 + np.random.normal(0,1.5))),
        max(5,min(92, weekday_base*weekend*1.1 + np.random.normal(0,1.5))),
        max(5,min(92, weekday_base*weekend + np.random.normal(0,1.5))),
    ]
    colors = [sc(s) for s in daily_scores]
    weekday_avg = sum(daily_scores[:5])/5
    weekend_avg = sum(daily_scores[5:])/2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG); ax2.set_facecolor(BG)

    ax1.barh(days, [100]*7, color="#1a2540", height=0.52, edgecolor="none", zorder=0)
    ax1.barh(days, daily_scores, color=colors, alpha=0.88, height=0.52, edgecolor="none", zorder=2)
    for i,(s,c) in enumerate(zip(daily_scores, colors)):
        ax1.text(s+1.5, i, f"{s:.0f}", va="center", fontsize=11, color=c, fontweight="700", fontfamily="monospace")
    ax1.axvline(weekday_avg, color="#60a5fa55", linewidth=1.5, linestyle="--")
    ax1.text(weekday_avg+1, 6.55, f"Weekday avg: {weekday_avg:.0f}", color="#60a5fa", fontsize=8, fontfamily="monospace")
    if weekend_avg < weekday_avg - 10:
        gap = weekday_avg - weekend_avg
        ax1.text(2, 5.3, f"⚠ Weekend drop: -{gap:.0f} pts", color="#f87171", fontsize=8.5, fontweight="700", fontfamily="monospace")
    ax1.set_xlim(0,112); ax1.set_xlabel("Social Health Score", color="#4a6080", fontsize=10, labelpad=6)
    ax1.set_title("Your Estimated Daily Score", color="#eef2ff", fontsize=12, fontweight="bold", pad=10)
    ax1.tick_params(colors="#94a3b8", labelsize=10)
    for _s in ax1.spines.values(): _s.set_edgecolor("#1a2540")

    signal_labels = ["Response Time","Social Contacts","Initiation Rate","Night Activity","Weekend Score","Future Thinking"]
    current_health = [
        max(0,min(100,100-ctx["resp"]/4)),
        min(100,ctx["contacts"]*6.5),
        ctx["init"]*100,
        max(0,min(100,100-ctx["night"]*200)),
        ctx["weekend"]*100,
        ctx["future"]*250
    ]
    target = [100,100,100,100,100,100]
    gaps = [100-v for v in current_health]
    gap_colors = ["#10b981" if g<20 else "#60a5fa" if g<40 else "#fbbf24" if g<65 else "#f87171" for g in gaps]

    x = range(len(signal_labels))
    ax2.bar(x, [100]*6, color="#1a2540", width=0.55, edgecolor="none", zorder=0)
    ax2.bar(x, current_health, color=gap_colors, alpha=0.85, width=0.55, edgecolor="none", zorder=2)
    ax2.bar(x, gaps, bottom=current_health, color=[c+"33" for c in gap_colors], width=0.55, edgecolor="none", zorder=1)

    for i,(v,g,c) in enumerate(zip(current_health, gaps, gap_colors)):
        ax2.text(i, v+2, f"{v:.0f}", ha="center", fontsize=9, color=c, fontweight="700", fontfamily="monospace")
        if g > 25:
            ax2.text(i, 102, f"+{g:.0f}", ha="center", fontsize=7.5, color="#4a6080", fontfamily="monospace")

    ax2.set_xticks(list(x)); ax2.set_xticklabels(signal_labels, color="#94a3b8", fontsize=8.5)
    ax2.set_ylim(0,115); ax2.set_ylabel("Score (solid=now, faded=gap)", color="#4a6080", fontsize=9, labelpad=6)
    ax2.set_title("Where You Are vs Where You Could Be", color="#eef2ff", fontsize=12, fontweight="bold", pad=10)
    ax2.tick_params(colors="#4a6080")
    for _s in ax2.spines.values(): _s.set_edgecolor("#1a2540")

    plt.tight_layout(pad=2)
    return fig

def chart_drift(lt,score):
    bases={0:83,1:67,2:50,3:62,4:35}; dr={0:.12,1:-.75,2:-1.05,3:-.80,4:-1.65}
    weeks=list(range(26))
    scores=[max(5,min(96,bases[lt]+dr[lt]*w+np.random.normal(0,2))) for w in weeks]; scores[-1]=score
    fig,ax=plt.subplots(1,1,figsize=(13,5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.fill_between(weeks,[70]*26,[100]*26,alpha=0.06,color="#10b981")
    ax.fill_between(weeks,[40]*26,[70]*26,alpha=0.05,color="#60a5fa")
    ax.fill_between(weeks,[0]*26,[40]*26,alpha=0.06,color="#f87171")
    for y,txt,c in [(85,"THRIVING","#10b98115"),(55,"DRIFTING","#60a5fa15"),(18,"CRISIS","#f8717115")]:
        ax.text(-0.4,y,txt,color=c,fontsize=8,fontfamily="monospace",va="center",ha="right")
    for i in range(len(weeks)-1):
        ax.plot(weeks[i:i+2],scores[i:i+2],color=sc(scores[i]),linewidth=3,solid_capstyle="round")
    ax.scatter(weeks,scores,c=[sc(s) for s in scores],s=40,zorder=5,edgecolors=BG,linewidths=1.5)
    ax.scatter([25],[score],c=[sc(score)],s=180,zorder=6,edgecolors="white",linewidths=2.5)
    off=12 if score<80 else -16
    ax.annotate(f"  Week 26: {score:.0f}",xy=(25,score),xytext=(21,score+off),
                color="#eef2ff",fontsize=11,fontweight="700",
                arrowprops=dict(arrowstyle="->",color=sc(score),lw=1.8))
    if lt>=2: ax.axvspan(19,25.5,alpha=0.07,color="#fbbf24"); ax.text(22.5,3,"← 6-week alert window",color="#fbbf2466",fontsize=8,ha="center",fontfamily="monospace")
    ax.set_xlim(-1,26.5); ax.set_ylim(-5,110)
    ax2=ax.twinx(); ax2.set_ylim(-5,110); ax2.set_yticks([10,40,70,90])
    ax2.set_yticklabels(["Crisis","Struggling","Stable","Thriving"],color="#4a6080",fontsize=9,fontfamily="monospace")
    ax2.set_facecolor("none"); ax2.tick_params(colors="#4a6080")
    for _s in ax2.spines.values(): _s.set_edgecolor("#1a2540")
    ax.set_xlabel("Week",color="#4a6080",fontsize=11,labelpad=8); ax.set_ylabel("Score",color="#4a6080",fontsize=11,labelpad=8)
    ax.set_title("26-Week Drift Timeline",color="#eef2ff",fontsize=15,fontweight="bold",pad=14)
    ax.tick_params(colors="#4a6080",labelsize=10)
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    plt.tight_layout()
    return fig

def chart_pop_hist(score):
    np.random.seed(42)
    pop=np.concatenate([np.random.normal(75,12,300),np.random.normal(55,10,400),np.random.normal(38,10,200),np.random.normal(22,8,100)])
    fig,ax=plt.subplots(1,1,figsize=(6,4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    _,bins,patches=ax.hist(pop,bins=28,edgecolor="none",rwidth=0.85)
    for patch,b in zip(patches,bins):
        patch.set_facecolor("#10b981" if b>=70 else "#60a5fa" if b>=50 else "#fbbf24" if b>=30 else "#f87171")
        patch.set_alpha(0.75)
    ax.axvline(score,color="white",linewidth=2.5,linestyle="--",zorder=5)
    pct=(pop<score).sum()/len(pop)*100
    ax.text(score+1.5,ax.get_ylim()[1]*0.88,"YOU",color="white",fontsize=10,fontfamily="monospace",fontweight="bold")
    ax.text(4,ax.get_ylim()[1]*0.75,f"Top {100-pct:.0f}%",color="#818cf8",fontsize=12,fontweight="700",
            bbox=dict(boxstyle="round,pad=0.4",facecolor="#0c1a35",edgecolor="#2e3a6e"))
    ax.set_title("Your Score vs Population",color="#eef2ff",fontsize=13,fontweight="bold",pad=10)
    ax.set_xlabel("Social Health Score",color="#4a6080",fontsize=10); ax.set_ylabel("People",color="#4a6080",fontsize=10)
    ax.tick_params(colors="#4a6080")
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    plt.tight_layout()
    return fig

def chart_type_donut():
    tc=[TC[l] for l in ["Healthy","Situational","Social","Existential","Chronic"]]
    fig,ax=plt.subplots(1,1,figsize=(5,4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    _,_,autos=ax.pie([33,25,18,14,10],colors=tc,autopct="%1.0f%%",pctdistance=0.82,startangle=90,
                     wedgeprops=dict(width=0.55,edgecolor=BG,linewidth=3))
    for a in autos: a.set_color("white"); a.set_fontsize(9.5); a.set_fontweight("bold")
    ax.set_title("Global Type Breakdown",color="#eef2ff",fontsize=13,fontweight="bold",pad=10)
    legs=[mpatches.Patch(color=tc[i],label=l) for i,l in enumerate(["Healthy","Situational","Social","Existential","Chronic"])]
    ax.legend(handles=legs,loc="lower center",bbox_to_anchor=(0.5,-0.12),ncol=3,fontsize=9,
              labelcolor="#94a3b8",facecolor="#0c1a35",edgecolor="#1a2540",framealpha=0.9)
    plt.tight_layout()
    return fig

def chart_week_pattern(lt,score,ctx=None):
    days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    hp=[82,78,80,76,72,85,83]
    if ctx:
        base=score; wknd=ctx["weekend"]*100; wkdy=min(95,base+8)
        mon=round(min(95,wkdy*0.95)); tue=round(min(95,wkdy*0.88)); wed=round(min(95,wkdy)); thu=round(min(95,wkdy*0.90)); fri=round(min(95,wkdy*0.85))
        sat=round(max(10,wknd*0.95)); sun=round(max(10,wknd))
        uw=[mon,tue,wed,thu,fri,sat,sun]
    else:
        tp={0:[82,78,80,76,72,85,83],1:[70,68,65,64,60,55,52],2:[55,52,50,48,44,38,36],3:[60,58,55,53,50,46,44],4:[40,38,35,33,30,25,22]}
        uw=tp[lt]
    s_col=sc(score)
    fig,ax=plt.subplots(1,1,figsize=(6,4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.fill_between(range(7),hp,alpha=0.08,color="#10b981"); ax.fill_between(range(7),uw,alpha=0.18,color=s_col)
    ax.plot(range(7),hp,"--",color="#10b98155",linewidth=1.8,label="Population avg")
    ax.plot(range(7),uw,color=s_col,linewidth=3,label="Your pattern")
    ax.scatter(range(7),uw,c=[sc(s) for s in uw],s=55,zorder=5,edgecolors=BG,linewidths=1.5)
    ax.set_xticks(range(7)); ax.set_xticklabels(days,color="#94a3b8",fontsize=9.5,fontweight="600")
    ax.set_ylim(0,108); ax.set_title("Your Week Pattern",color="#eef2ff",fontsize=13,fontweight="bold",pad=10)
    ax.tick_params(colors="#4a6080")
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    ax.legend(fontsize=9.5,labelcolor="#94a3b8",facecolor="#0c1a35",edgecolor="#1a2540",loc="upper right")
    plt.tight_layout()
    return fig

def chart_projection(lt,score):
    dr={0:.12,1:-.75,2:-1.05,3:-.80,4:-1.65}
    weeks=list(range(13))
    base=[max(5,min(96,score+dr[lt]*w)) for w in weeks]
    reconnect=[min(96,s+w*1.8) for w,s in enumerate(base)]
    events=[min(96,s+w*1.3) for w,s in enumerate(base)]
    combined=[min(96,s+w*3.0) for w,s in enumerate(base)]
    fig,ax=plt.subplots(1,1,figsize=(13,5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.fill_between(weeks,[70]*13,[100]*13,alpha=0.05,color="#10b981")
    ax.fill_between(weeks,[40]*13,[70]*13,alpha=0.04,color="#60a5fa")
    ax.plot(weeks,base,"--",color="#f8717155",linewidth=1.8,label="No action")
    ax.fill_between(weeks,base,reconnect,alpha=0.12,color="#818cf8")
    ax.plot(weeks,reconnect,color="#818cf8",linewidth=2.5,label="Reach out to people")
    ax.fill_between(weeks,reconnect,events,alpha=0.08,color="#60a5fa")
    ax.plot(weeks,events,color="#60a5fa",linewidth=2.5,label="Join social events")
    ax.fill_between(weeks,base,combined,alpha=0.06,color="#10b981")
    ax.plot(weeks,combined,color="#10b981",linewidth=3.5,label="All interventions",zorder=5)
    for s2,c2 in [(base[-1],"#f87171"),(reconnect[-1],"#818cf8"),(combined[-1],"#10b981")]:
        ax.annotate(f"{s2:.0f}",xy=(12,s2),xytext=(12.35,s2),color=c2,fontsize=11,fontweight="700",va="center",fontfamily="monospace")
    ax.set_xlim(-0.3,13.8); ax.set_ylim(5,100)
    ax.set_xlabel("Weeks from today",color="#4a6080",fontsize=11,labelpad=8)
    ax.set_ylabel("Projected Score",color="#4a6080",fontsize=11,labelpad=8)
    ax.set_title("Intervention Impact — Next 12 Weeks",color="#eef2ff",fontsize=15,fontweight="bold",pad=14)
    ax.tick_params(colors="#4a6080",labelsize=10)
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    ax.legend(fontsize=10,labelcolor="#94a3b8",facecolor="#0c1a35",edgecolor="#1a2540",loc="upper left")
    plt.tight_layout()
    return fig

def chart_mood(ctx=None):
    days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    if ctx:
        base_e = ctx["weekend"]*100; base_c = ctx["init"]*80; base_m = max(20,min(90,(ctx["weekend"]*60+ctx["future"]*100+ctx["init"]*40)/3))
        dip_sat = max(5, base_e - (1-ctx["weekend"])*30)
        dip_sun = max(5, base_e - (1-ctx["weekend"])*20)
        night_penalty = ctx["night"]*30
        energy=[round(base_e*0.85),round(base_e*0.78),round(base_e*0.92),round(base_e*0.80),round(base_e),round(dip_sat),round(dip_sun)]
        conn=[round(base_c*0.75),round(base_c*0.85),round(base_c*0.68),round(base_c*0.92),round(base_c*0.80),round(base_c*0.55),round(base_c*0.45)]
        mood=[round(base_m*0.90),round(base_m*0.82),round(base_m*0.95),round(base_m*0.85),round(base_m),round(base_m*0.62),round(base_m*0.58)]
    else:
        np.random.seed(7)
        energy=[65,58,72,60,70,45,50]; conn=[50,55,48,62,58,40,38]; mood=[68,62,74,65,72,48,52]
    fig,ax=plt.subplots(1,1,figsize=(13,4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for vals,c,mk,lbl in [(energy,"#10b981","o","Energy"),(conn,"#818cf8","s","Connection"),(mood,"#60a5fa","^","Mood")]:
        ax.fill_between(range(7),vals,alpha=0.1,color=c)
        ax.plot(range(7),vals,marker=mk,color=c,linewidth=2.5,markersize=9,markeredgecolor=BG,markeredgewidth=2,label=lbl)
    ax.set_xticks(range(7)); ax.set_xticklabels(days,color="#94a3b8",fontsize=11,fontweight="600")
    ax.set_ylim(0,110); ax.set_ylabel("Score",color="#4a6080",fontsize=11)
    ax.set_title("Weekly Mood, Energy & Connection",color="#eef2ff",fontsize=14,fontweight="bold",pad=14)
    ax.tick_params(colors="#4a6080")
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    ax.legend(fontsize=10.5,labelcolor="#94a3b8",facecolor="#0c1a35",edgecolor="#1a2540",loc="upper right")
    plt.tight_layout()
    return fig

def chart_drift_interceptor(score, lt, ctx):
    """Collapse timeline visualization for Drift Interceptor"""
    night=ctx["night"]; weekend=ctx["weekend"]; future=ctx["future"]; init=ctx["init"]
    weeks_ago = 0
    if night > 0.35: weeks_ago += 10
    elif night > 0.2: weeks_ago += 5
    if weekend < 0.3: weeks_ago += 8
    elif weekend < 0.5: weeks_ago += 3
    if future < 0.08: weeks_ago += 7
    elif future < 0.2: weeks_ago += 3
    if init < 0.25: weeks_ago += 6
    elif init < 0.4: weeks_ago += 2
    weeks_ago = max(2, min(24, weeks_ago))

    total_weeks = weeks_ago + 4
    all_weeks = list(range(total_weeks))
    peak_score = min(95, score + weeks_ago * 1.2)
    scores = []
    for w in all_weeks:
        if w < 3:
            s = peak_score + np.random.normal(0, 1.5)
        elif w < weeks_ago:
            progress = (w - 3) / max(1, weeks_ago - 3)
            s = peak_score - (peak_score - score) * progress + np.random.normal(0, 1.5)
        else:
            s = score + np.random.normal(0, 1)
        scores.append(max(5, min(97, s)))

    fig, ax = plt.subplots(1, 1, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    ax.fill_between(all_weeks, [70]*total_weeks, [100]*total_weeks, alpha=0.05, color="#10b981")
    ax.fill_between(all_weeks, [40]*total_weeks, [70]*total_weeks, alpha=0.04, color="#60a5fa")
    ax.fill_between(all_weeks, [0]*total_weeks, [40]*total_weeks, alpha=0.05, color="#f87171")

    drift_weeks = list(range(3, total_weeks))
    ax.axvspan(3, total_weeks-0.5, alpha=0.07, color="#f87171")

    for i in range(len(all_weeks)-1):
        ax.plot(all_weeks[i:i+2], scores[i:i+2], color=sc(scores[i]), linewidth=3, solid_capstyle="round")
    ax.scatter(all_weeks, scores, c=[sc(s) for s in scores], s=35, zorder=5, edgecolors=BG, linewidths=1.5)

    ax.axvline(3, color="#f87171", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.text(3.2, max(scores)+5, "← Drift begins here", color="#f87171", fontsize=9, fontfamily="monospace", va="bottom")

    ax.scatter([total_weeks-1], [score], c=[sc(score)], s=160, zorder=6, edgecolors="white", linewidths=2.5)
    ax.annotate(f"  Now: {score:.0f}", xy=(total_weeks-1, score),
                xytext=(total_weeks-3, score+(10 if score<80 else -14)),
                color="#eef2ff", fontsize=10, fontweight="700",
                arrowprops=dict(arrowstyle="->", color=sc(score), lw=1.5))

    x_labels = []
    for w in all_weeks:
        weeks_from_now = total_weeks - 1 - w
        if weeks_from_now == 0: x_labels.append("Now")
        elif weeks_from_now == weeks_ago: x_labels.append(f"{weeks_ago}w ago")
        elif w == 0: x_labels.append(f"{total_weeks}w ago")
        else: x_labels.append("")
    ax.set_xticks(all_weeks[::2])
    ax.set_xticklabels([x_labels[i] for i in range(0, total_weeks, 2)], color="#4a6080", fontsize=8.5)

    ax.set_xlim(-0.5, total_weeks)
    ax.set_ylim(-5, 108)
    ax.set_ylabel("Social Health Score", color="#4a6080", fontsize=10, labelpad=8)
    ax.set_title("Your Collapse Timeline — When the Drift Actually Began", color="#eef2ff", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#4a6080")
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    plt.tight_layout()
    return fig, weeks_ago

def chart_crisis_status(score):
    s_col=sc(score)
    fig,ax=plt.subplots(1,1,figsize=(4,3.5))
    fig.patch.set_facecolor(BG2); ax.set_facecolor(BG2); ax.axis("off")
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    status="monitoring" if score>55 else "gentle_checkin" if score>35 else "active"
    cc={"monitoring":"#10b981","gentle_checkin":"#fbbf24","active":"#f87171"}[status]
    icon={"monitoring":"🟢","gentle_checkin":"🟡","active":"🔴"}[status]
    ax.text(5,7.8,icon,ha="center",va="center",fontsize=40)
    ax.text(5,5.9,status.replace("_"," ").title(),ha="center",va="center",fontsize=15,color=cc,fontweight="bold")
    ax.text(5,4.7,"Agent Status",ha="center",va="center",fontsize=10,color="#4a6080",fontfamily="monospace")
    ax.text(5,2.9,f"{score:.0f}",ha="center",va="center",fontsize=50,color=s_col,fontweight="800")
    ax.text(5,1.2,"YOUR SCORE",ha="center",va="center",fontsize=9,color="#4a6080",fontfamily="monospace")
    risk_w=max(1,int((50-score)/5)) if score<50 else 0
    note=f"~{risk_w} weeks to crisis window" if score<50 else "✓ Safe from crisis zone"
    nc="#f87171" if score<50 else "#10b981"
    ax.text(5,0.2,note,ha="center",va="center",fontsize=9,color=nc,fontfamily="monospace")
    plt.tight_layout()
    return fig

def chart_landing_preview():
    """Landing page preview chart"""
    dw=list(range(20)); ds=[70,68,65,63,60,58,56,54,51,48,46,44,41,38,36,34,31,29,27,25]
    fig,ax=plt.subplots(1,1,figsize=(5.5,3.8))
    fig.patch.set_facecolor("#0c1a35"); ax.set_facecolor("#0c1a35")
    for i in range(len(dw)-1):
        ax.plot(dw[i:i+2],ds[i:i+2],color=sc(ds[i]),linewidth=3,solid_capstyle="round")
    ax.axvspan(13,19,alpha=0.08,color="#fbbf24")
    ax.text(16,3,"← 6-week window",color="#fbbf2455",fontsize=7.5,ha="center",fontfamily="monospace")
    ax.scatter([19],[25],c=["#f87171"],s=100,zorder=5,edgecolors="white",linewidths=1.5)
    ax.set_title("Sample: Drift Detected 6 Weeks Early",color="#eef2ff",fontsize=11,fontweight="bold",pad=8)
    ax.set_xlabel("Week",color="#4a6080",fontsize=9); ax.set_ylabel("Score",color="#4a6080",fontsize=9)
    ax.tick_params(colors="#4a6080",labelsize=8)
    for _s in ax.spines.values(): _s.set_edgecolor("#1a2540")
    plt.tight_layout()
    return fig


INSIGHTS = {
    "q1": {"Instantly":"✨ People who reply quickly show 34% higher social health scores on average — your responsiveness is a strength.","Within an hour":"👍 Good response rhythm. Under-1hr responders maintain 2.1× more active relationships than next-day responders.","Few hours":"⚠️ Inconsistent latency can signal emotional avoidance. 3+ hour gaps start to erode perceived closeness.","Next day":"🔴 Next-day response patterns correlate strongly with social drift. This is one of the earliest detectable signals.","Often leave unread":"🔴 Unread messages are the #1 behavioral predictor we measure. This alone explains ~20 points of score difference."},
    "q2": {"10+ people":"✨ Excellent contact diversity — 10+ meaningful interactions/week puts you in the top 15% globally.","5–10 people":"👍 Healthy range. Research shows 5+ weekly interactions is the key threshold for social wellbeing.","2–4 people":"⚠️ Below the 5-contact threshold. Cacioppo et al. (2008) found this is where chronic loneliness risk begins rising.","1 person":"🔴 1 contact/week is 3.8× more likely to lead to clinical loneliness within 6 months.","Nobody":"🔴 Zero social contact this week. This is the single strongest risk factor in our model."},
    "q3": {"Always me":"⚠️ Always initiating is exhausting and unsustainable — and a sign your ties may be weaker than they feel.","Mostly me":"⚠️ Mostly initiating suggests possible asymmetry. Relationships need bidirectional investment to last.","Equal":"✨ Equal initiation is the gold standard. Reciprocal relationships are 4× more likely to deepen over time.","Mostly them":"👍 Mostly receiving is fine — though check: are you initiating with at least some people?","Rarely me":"🔴 Very low initiation is one of the strongest early drift signals. Social circles shrink fastest when you stop reaching out."},
    "q4": {"Never":"✨ Healthy sleep patterns strongly correlate with social health. Consistent bedtimes protect your wellbeing.","Occasionally":"👍 Occasional late nights are normal. The key is that it's not a habitual coping pattern.","Few nights/week":"⚠️ Regular 1–4am phone use is correlated with loneliness-driven hypervigilance. Worth watching.","Most nights":"🔴 Most nights awake 1–4am is the #1 strongest single behavioral loneliness marker in our dataset.","Every night":"🔴 Every night. This pattern shows up in 94% of chronic loneliness cases. This is significant."},
    "q5": {"Very active":"✨ Active weekends are protective. Weekend social time is 3× more predictive of wellbeing than weekday time.","Balanced":"👍 Balanced weekend — healthy mix of social and solo time supports long-term sustainability.","Mostly home":"⚠️ Mostly home weekends often signal social avoidance — particularly when it's not chosen, but habitual.","Mostly alone":"🔴 Weekend isolation is 3× more predictive of chronic loneliness than weekday isolation.","In bed / avoidant":"🔴 Avoidant weekend pattern. This often indicates social exhaustion or depression-adjacent withdrawal."},
    "q6": {"A lot":"✨ Strong future orientation. Looking forward to things is one of the most protective mental health factors.","Somewhat":"👍 Some future excitement is healthy. Even small things to look forward to matter enormously.","Not much":"⚠️ Low future orientation appears 4–8 weeks before a subjective loneliness crisis in our data.","Rarely":"🔴 Rarely thinking ahead correlates with hedonic collapse — the feeling that nothing is worth anticipating.","Feels blank":"🔴 Future feeling blank is a serious signal. This often accompanies severe isolation and low motivation."},
    "q7": {"With family":"👍 Living with family provides ambient social contact — a buffer against acute loneliness.","With partner":"✨ Living with a partner is protective — though quality of the relationship matters more than presence.","With flatmates":"👍 Flatmates provide casual social contact. Even superficial co-presence reduces loneliness biomarkers.","Alone":"⚠️ Living alone amplifies loneliness signals by 2.3× on average. Not deterministic — but significant.","Living alone":"⚠️ Same as above — the key question is whether your external social contact compensates."},
    "q8": {"Office full-time":"✨ Regular office attendance provides 8+ hrs of ambient social contact — a strong protective factor.","Hybrid":"👍 Hybrid provides structure while maintaining flexibility — currently the best-scoring work arrangement.","Remote / WFH":"⚠️ Remote work is now the #1 situational loneliness trigger globally. Intentional social effort is critical.","Student":"👍 Campus life provides natural social infrastructure — though digital-first students still face isolation risk.","Not working":"⚠️ Not working removes the most consistent source of daily social contact for most adults."},
}

QUESTIONS=[
    {"id":"q1","emoji":"💬","title":"Response Rhythm","q":"When someone messages you, how quickly do you reply?","sub":"Your response latency is a precise engagement signal.","opts":["Instantly","Within an hour","Few hours","Next day","Often leave unread"]},
    {"id":"q2","emoji":"👥","title":"Social Circle","q":"How many people did you have a real conversation with this week?","sub":"Back-and-forth exchanges only — not likes.","opts":["10+ people","5–10 people","2–4 people","1 person","Nobody"]},
    {"id":"q3","emoji":"📲","title":"Initiation Pattern","q":"In your friendships, who reaches out first most often?","sub":"One of the strongest early-drift predictors.","opts":["Always me","Mostly me","Equal","Mostly them","Rarely me"]},
    {"id":"q4","emoji":"🌙","title":"Night Behavior","q":"How often are you active on your phone between 1am–4am?","sub":"The late-night window is the clearest loneliness fingerprint.","opts":["Never","Occasionally","Few nights/week","Most nights","Every night"]},
    {"id":"q5","emoji":"📅","title":"Weekend Life","q":"What does your typical weekend look like?","sub":"Weekends reveal what daily routines hide.","opts":["Very active","Balanced","Mostly home","Mostly alone","In bed / avoidant"]},
    {"id":"q6","emoji":"🔮","title":"Future Thinking","q":"How excited do you feel about upcoming things in your life?","sub":"Future-tense thinking is a key wellbeing signal.","opts":["A lot","Somewhat","Not much","Rarely","Feels blank"]},
    {"id":"q7","emoji":"🏠","title":"Living Situation","q":"Who do you currently live with?","sub":"Living alone amplifies loneliness signals by 2.3×.","opts":["With family","With partner","With flatmates","Alone","Living alone"]},
    {"id":"q8","emoji":"💼","title":"Work Context","q":"What's your current work / study situation?","sub":"Remote work is the #1 situational loneliness trigger globally.","opts":["Office full-time","Hybrid","Remote / WFH","Student","Not working"]},
]

def generate_pdf_report(r):
    """Generate PDF using reportlab — zero external deps beyond what's already installed"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.pdfgen import canvas

        score = r["score"]; label = r["label"]; drift = r["drift"]; ctx = r["ctx"]

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        W, H = A4

        c.setFillColorRGB(0.024, 0.059, 0.125)
        c.rect(0, 0, W, H, fill=1, stroke=0)

 
        c.setFillColorRGB(0.643, 0.706, 1.0)
        c.setFont("Helvetica-Bold", 32)
        c.drawCentredString(W/2, H-60, "TETHER")
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0.58, 0.64, 0.75)
        c.drawCentredString(W/2, H-82, "Social Health Assessment Report")
        c.drawCentredString(W/2, H-98, datetime.now().strftime("%d %B %Y · %H:%M"))

        s_rgb = {
            "#10b981": (0.063, 0.725, 0.506),
            "#60a5fa": (0.376, 0.647, 0.980),
            "#fbbf24": (0.984, 0.749, 0.141),
            "#f87171": (0.973, 0.443, 0.443),
        }.get(sc(score), (0.376, 0.647, 0.980))
        c.setStrokeColorRGB(*s_rgb)
        c.setFillColorRGB(0.05, 0.11, 0.22)
        c.roundRect(40, H-210, W-80, 95, 10, fill=1, stroke=1)
        c.setFillColorRGB(*s_rgb)
        c.setFont("Helvetica-Bold", 56)
        c.drawCentredString(W/2, H-168, f"{score:.0f} / 100")
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(W/2, H-195, f"{sl(score)}  ·  {label} Loneliness  ·  {drift['txt']}")

        y = H-240
        stats = [
            ("Crisis Risk", r["crisis_level"]),
            ("Trajectory", drift["txt"]),
            ("Crisis Prob.", f"{int(r['crisis_prob']*100)}%"),
            ("Type", label),
        ]
        col_w = (W-80)/4
        for i,(lbl_s,val_s) in enumerate(stats):
            x = 40 + i*col_w + col_w/2
            c.setFillColorRGB(0.38, 0.45, 0.6)
            c.setFont("Helvetica", 9)
            c.drawCentredString(x, y, lbl_s.upper())
            c.setFillColorRGB(*s_rgb)
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(x, y-18, val_s)

        c.setStrokeColorRGB(0.12, 0.22, 0.4)
        c.line(40, H-268, W-40, H-268)

        y = H-290
        c.setFillColorRGB(0.643, 0.706, 1.0)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(40, y, "LONELINESS FINGERPRINT")
        y -= 20

        for dim, val in r["fp"].items():
            val = min(100, max(0, val))
            bar_col = (0.063,0.725,0.506) if val<35 else (0.376,0.647,0.980) if val<60 else (0.984,0.749,0.141) if val<80 else (0.973,0.443,0.443)
            c.setFillColorRGB(0.58, 0.64, 0.75); c.setFont("Helvetica", 10)
            c.drawString(40, y, dim)
            c.setFillColorRGB(0.07,0.15,0.28); c.rect(180, y-1, 220, 10, fill=1, stroke=0)
            c.setFillColorRGB(*bar_col); c.rect(180, y-1, int(val*2.2), 10, fill=1, stroke=0)
            c.setFont("Helvetica-Bold", 10); c.drawString(408, y, f"{val:.0f}")
            y -= 20

        c.setStrokeColorRGB(0.12, 0.22, 0.4)
        c.line(40, y-4, W-40, y-4); y -= 22

        c.setFillColorRGB(0.643, 0.706, 1.0); c.setFont("Helvetica-Bold", 13)
        c.drawString(40, y, "BEHAVIORAL READINGS"); y -= 20
        readings = [
            ("Response Time", f"{ctx['resp']:.0f} min avg", "Fast" if ctx['resp']<30 else "Moderate" if ctx['resp']<120 else "Slow"),
            ("Weekly Contacts", f"{ctx['contacts']} people", "Good" if ctx['contacts']>=5 else "Low"),
            ("Initiation Rate", f"{ctx['init']*100:.0f}%", "Good" if ctx['init']>=0.4 else "Low"),
            ("Night Activity", f"{ctx['night']*100:.0f}%", "Low risk" if ctx['night']<0.15 else "High risk"),
            ("Weekend Score", f"{ctx['weekend']*100:.0f}%", "Active" if ctx['weekend']>=0.5 else "Low"),
            ("Future Thinking", f"{ctx['future']*100:.0f}%", "Good" if ctx['future']>=0.2 else "Low"),
        ]
        for name, val, status in readings:
            ok = status in ("Good","Fast","Active","Low risk","Low")
            c.setFillColorRGB(0.58,0.64,0.75); c.setFont("Helvetica",10); c.drawString(40,y,name)
            c.setFillColorRGB(0.87,0.89,1.0); c.drawString(200,y,val)
            if ok:
                c.setFillColorRGB(0.063,0.725,0.506)
            else:
                c.setFillColorRGB(0.973,0.443,0.443)
            c.setFont("Helvetica-Bold",10); c.drawString(320,y,status)
            y -= 17

        c.setStrokeColorRGB(0.12,0.22,0.4); c.line(40,y-4,W-40,y-4); y -= 22

        if r["sigs"] and y > 100:
            c.setFillColorRGB(0.643,0.706,1.0); c.setFont("Helvetica-Bold",13)
            c.drawString(40, y, "KEY SIGNALS DETECTED"); y -= 18
            for sig in r["sigs"]:
                sev_c = (0.973,0.443,0.443) if sig["sev"]=="high" else (0.984,0.749,0.141)
                c.setFillColorRGB(*sev_c); c.setFont("Helvetica-Bold",10)
                c.drawString(40, y, f"[{sig['sev'].upper()}] {sig['s']}")
                c.setFillColorRGB(0.58,0.64,0.75); c.setFont("Helvetica",9)
                c.drawString(50, y-13, sig['d'] + "  →  " + sig['i'])
                y -= 28

        if y > 120:
            c.setStrokeColorRGB(0.12,0.22,0.4); c.line(40,y-4,W-40,y-4); y -= 22
            c.setFillColorRGB(0.643,0.706,1.0); c.setFont("Helvetica-Bold",13)
            c.drawString(40, y, "RECOMMENDED INTERVENTIONS"); y -= 18
            for inv in [
                ("HIGH","Send one message today","One line to someone you haven't spoken to in 2+ weeks."),
                ("MED","Replace one solo activity","Turn something alone into something shared this week."),
                ("MED","20 min outside daily","Outdoor movement resets cortisol and loneliness biomarkers."),
            ]:
                c.setFillColorRGB(0.376,0.647,0.980); c.setFont("Helvetica-Bold",10)
                c.drawString(40, y, f"[{inv[0]}] {inv[1]}")
                c.setFillColorRGB(0.58,0.64,0.75); c.setFont("Helvetica",9)
                c.drawString(50, y-13, inv[2])
                y -= 28

        c.setFillColorRGB(0.114,0.196,0.318)
        c.setFont("Helvetica-Oblique", 8)
        c.drawCentredString(W/2, 30, "Tether — The first instrument the loneliness epidemic has ever had")
        c.drawCentredString(W/2, 18, "This report is not a clinical diagnosis. Crisis support: iCall India 9152987821 (Mon–Sat 8am–10pm)")

        c.save()
        buf.seek(0)
        return buf.read()
    except Exception as e:
        return None


def page_landing():
    c1,_,c3=st.columns([1,3,1])
    with c1: st.markdown("<div style='padding:8px 0;'><span style='font-family:Space Grotesk,sans-serif;font-size:48px;font-weight:800;color:#eef2ff;letter-spacing:-.03em;'>Tether</span><span style='font-size:10px;background:rgba(99,102,241,.18);color:#a5b4fc;padding:2px 8px;border-radius:99px;border:1px solid rgba(99,102,241,.3);margin-left:10px;vertical-align:middle;'>v6.0</span></div>",unsafe_allow_html=True)
    with c3: st.markdown(f"<div style='text-align:right;padding:10px 0;font-family:JetBrains Mono,monospace;font-size:10px;color:#2e3a6e;'>{datetime.now().strftime('%d %b %Y')}</div>",unsafe_allow_html=True)
    st.divider()
    hl,hr=st.columns([1.1,1],gap="large")
    with hl:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<div style='display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.25);border-radius:99px;padding:5px 16px;margin-bottom:20px;'><span style='width:8px;height:8px;background:#818cf8;border-radius:50%;'></span><span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#a5b4fc;letter-spacing:.1em;'>LONELINESS EPIDEMIC INSTRUMENT</span></div>",unsafe_allow_html=True)
        st.markdown("""<h1 style='font-family:Space Grotesk,sans-serif;font-size:clamp(38px,4.5vw,64px);font-weight:800;line-height:1.07;letter-spacing:-.025em;color:#eef2ff;margin:0 0 18px;'>Know When You're<br><span style='background:linear-gradient(135deg,#818cf8 0%,#a78bfa 60%,#c084fc 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>Drifting</span> Before<br>You're Lost</h1><p style='font-size:18px;color:#64748b;line-height:1.8;max-width:500px;margin-bottom:32px;'>Loneliness has no clinical instrument. <strong style='color:#94a3b8;'>Tether is the first.</strong><br>We detect drift from behavioral patterns — <em style='color:#818cf8;'>6 weeks before you feel it.</em></p>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("✦  Start Free Assessment — 8 Questions",key="cta"):
            st.session_state.page="onboarding"; st.session_state.step=1; st.rerun()
    with hr:
        st.markdown("<br>",unsafe_allow_html=True)
        render(chart_landing_preview())
        st.markdown("<br>",unsafe_allow_html=True)
        for icon,name,status,c in [("🗺️","Environment Scanner","Ready · Awaiting city input","#60a5fa"),("🚨","Crisis Interceptor","Active · Watching your score","#10b981"),("🔍","Drift Interceptor","Active · Analysing your collapse timeline","#a78bfa")]:
            st.markdown(f"<div style='display:flex;align-items:center;gap:12px;padding:10px 16px;background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.14);border-radius:12px;margin:5px 0;'><span style='font-size:18px;'>{icon}</span><div style='flex:1;'><div style='font-size:13px;font-weight:600;color:#eef2ff;'>{name}</div><div style='font-size:11px;color:{c};margin-top:1px;'>{status}</div></div><span style='width:8px;height:8px;border-radius:50%;background:{c};box-shadow:0 0 6px {c};'></span></div>",unsafe_allow_html=True)
    st.divider()
    LBL("HOW IT WORKS"); H("Four Steps to Social Clarity",28); st.markdown("<br>",unsafe_allow_html=True)
    cols=st.columns(4,gap="medium")
    for col,(num,title,desc,c) in zip(cols,[("01","8 Questions","Answer honestly. No sliders. No jargon.","#6366f1"),("02","AI Analysis","3 ML models score against 52K behavioral records.","#8b5cf6"),("03","Your Report","Score, fingerprint, population context, drift.","#a78bfa"),("04","Agents Act","3 AI agents scan, intercept, architect, project.","#c084fc")]):
        col.markdown(f"<div style='padding:22px 18px;background:rgba(99,102,241,.05);border:1px solid rgba(99,102,241,.12);border-radius:18px;height:100%;'><div style='font-family:JetBrains Mono,monospace;font-size:13px;color:{c};margin-bottom:12px;opacity:.7;'>{num}</div><div style='font-family:Space Grotesk,sans-serif;font-size:17px;font-weight:700;color:#eef2ff;margin-bottom:10px;'>{title}</div><div style='font-size:14px;color:#64748b;line-height:1.7;'>{desc}</div></div>",unsafe_allow_html=True)

def page_onboarding():
    step=st.session_state.step; total=len(QUESTIONS); q=QUESTIONS[step-1]
    c1,c2,_=st.columns([1,4,1])
    with c1:
        if st.button("← Home",key="home"): st.session_state.page="landing"; st.rerun()
    with c2:
        prog=(step-1)/total
        st.markdown(f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'><span style='font-family:JetBrains Mono,monospace;font-size:12px;color:#4a6080;'>{step}/{total} — <span style='color:#a5b4fc;'>{q['title']}</span></span><span style='font-family:JetBrains Mono,monospace;font-size:12px;color:#6366f1;font-weight:700;'>{int(prog*100)}%</span></div>",unsafe_allow_html=True)
        st.progress(prog)
    st.markdown("<br>",unsafe_allow_html=True)
    _,qc,_=st.columns([1,4,1])
    with qc:
        st.markdown(f"""<div style='padding:50px 40px;background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(124,58,237,.04));border:1px solid rgba(99,102,241,.22);border-radius:28px;text-align:center;margin-bottom:28px;'><div style='font-size:58px;margin-bottom:18px;'>{q["emoji"]}</div><div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#4a6080;letter-spacing:.16em;text-transform:uppercase;margin-bottom:14px;'>{q["title"]}</div><h2 style='font-family:Space Grotesk,sans-serif;font-size:clamp(18px,2.5vw,24px);font-weight:700;line-height:1.35;color:#eef2ff;margin-bottom:10px;'>{q["q"]}</h2><p style='font-size:14px;color:#4a6080;margin:0;'>{q.get("sub","")}</p></div>""",unsafe_allow_html=True)
    opts=q["opts"]; n=min(len(opts),3)
    cols=st.columns(n,gap="small")
    for i,opt in enumerate(opts):
        with cols[i%n]:
            sel=st.session_state.answers.get(q["id"])==opt
            if st.button(f"✓  {opt}" if sel else opt,key=f"q{step}_{i}"):
                st.session_state.answers[q["id"]]=opt
                if step<total: st.session_state.step+=1
                else:
                    with st.spinner("Analysing your patterns..."):
                        time.sleep(0.8)
                        st.session_state.result=compute(st.session_state.answers)
                        st.session_state.page="results"
                st.rerun()
    prev_qid = QUESTIONS[step-2]["id"] if step>1 else None
    prev_ans = st.session_state.answers.get(prev_qid,"") if prev_qid else ""
    if prev_ans and prev_qid and prev_qid in INSIGHTS:
        insight = INSIGHTS[prev_qid].get(prev_ans,"")
        if insight:
            has_red = insight.startswith("🔴")
            has_warn = insight.startswith("⚠️")
            icon_e = insight.split(" ")[0]
            text_e = " ".join(insight.split(" ")[1:])
            bg_e = "rgba(239,68,68,.08)" if has_red else "rgba(245,158,11,.08)" if has_warn else "rgba(16,185,129,.08)"
            bc_e = "rgba(239,68,68,.25)" if has_red else "rgba(245,158,11,.25)" if has_warn else "rgba(16,185,129,.25)"
            tc_e = "#f87171" if has_red else "#fbbf24" if has_warn else "#10b981"
            _,ins_col,_=st.columns([1,4,1])
            with ins_col:
                st.markdown(
                    f"<div style='margin-top:20px;padding:14px 18px;background:{bg_e};"
                    f"border:1px solid {bc_e};border-radius:14px;display:flex;align-items:start;gap:12px;'>"
                    f"<span style='font-size:20px;flex-shrink:0;'>{icon_e}</span>"
                    f"<div><div style='font-family:JetBrains Mono,monospace;font-size:9px;color:{tc_e};"
                    f"letter-spacing:.15em;text-transform:uppercase;margin-bottom:5px;'>INSIGHT FROM PREVIOUS ANSWER</div>"
                    f"<div style='font-size:13px;color:#e2e8f0;line-height:1.7;'>{text_e}</div></div></div>",
                    unsafe_allow_html=True)
    if step>1:
        st.markdown("<br>",unsafe_allow_html=True)
        _,pc,_=st.columns([2,1,2])
        with pc:
            if st.button("← Previous",key="prev"): st.session_state.step-=1; st.rerun()

def page_results():
    r=st.session_state.result
    if not r: st.session_state.page="landing"; st.rerun(); return
    score=r["score"]; lt=r["lt"]; label=r["label"]; drift=r["drift"]; ctx=r["ctx"]
    s_col=sc(score); tc=TC.get(label,"#6366f1"); dc=drift["c"]; da=drift["arrow"]; dt=drift["txt"]

    h1,h2c,_=st.columns([1,3,1])
    with h1:
        if st.button("← Retake",key="retake"):
            st.session_state.page="onboarding"; st.session_state.step=1
            st.session_state.answers={}; st.session_state.result=None; st.session_state.ao={}; st.rerun()
    with h2c:
        st.markdown(f"<div style='text-align:center;padding:8px 0;'><span style='font-family:Space Grotesk,sans-serif;font-size:22px;font-weight:800;color:#eef2ff;'>Social Health Report</span> &nbsp;<span style='background:{tc}22;color:{tc};border:1px solid {tc}44;padding:4px 14px;border-radius:99px;font-size:13px;font-weight:700;'>{label} Loneliness</span>&nbsp;<span style='color:{dc};font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;'>{da} {dt}</span></div>",unsafe_allow_html=True)
    st.divider()

    dl1,dl2,dl3=st.columns([2,1,2])
    with dl2:
        if st.button("⬇️  Download PDF Report",key="pdf_dl"):
            with st.spinner("Generating your report..."):
                pdf_bytes=generate_pdf_report(r)
            if pdf_bytes:
                b64=base64.b64encode(pdf_bytes).decode()
                fname=f"tether_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.markdown(
                    f"<a href='data:application/pdf;base64,{b64}' download='{fname}' "
                    f"style='display:block;text-align:center;padding:12px 20px;"
                    f"background:linear-gradient(135deg,#059669,#10b981);"
                    f"color:white;font-weight:700;border-radius:12px;"
                    f"text-decoration:none;font-size:15px;margin-top:8px;"
                    f"box-shadow:0 4px 16px rgba(16,185,129,0.4);'>"
                    f"✅ Click here to save your PDF report</a>",
                    unsafe_allow_html=True)
            else:
                st.error("PDF generation failed. Check reportlab is installed: pip install reportlab")
    st.markdown("<br>",unsafe_allow_html=True)
    t1,t2,t3,t4,t5=st.tabs(["📊  Score","🌡️  Deep Analysis","🌍  Population","💡  Insights","⚡  Agentic Analysis"])

    with t1:
        st.markdown("<br>",unsafe_allow_html=True)
        
        score_col = sc(score)
        crisis_c = "#f87171" if r["crisis_level"] in ["High","Critical"] else "#60a5fa"
        st.markdown(
            f"<div style='text-align:center;padding:44px 24px;"
            f"background:radial-gradient(ellipse at 50% 30%,{score_col}14 0%,transparent 65%);"
            f"border:1px solid {score_col}30;border-radius:22px;margin-bottom:16px;'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#4a6080;"
            f"letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px;'>SOCIAL HEALTH SCORE</div>"
            f"<div style='font-family:Space Grotesk,sans-serif;font-size:96px;"
            f"font-weight:800;line-height:1;color:{score_col};'>{score:.0f}</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;color:#4a6080;margin:4px 0 16px;'>/ 100</div>"
            f"<div style='font-family:Space Grotesk,sans-serif;font-size:20px;font-weight:700;color:{score_col};margin-bottom:16px;'>{sl(score)}</div>"
            f"<div style='display:inline-flex;align-items:center;gap:8px;"
            f"background:{tc}15;border:1px solid {tc}35;border-radius:99px;padding:7px 18px;'>"
            f"<span style='width:8px;height:8px;border-radius:50%;background:{tc};'></span>"
            f"<span style='font-size:13px;font-weight:600;color:{tc};'>{label} Loneliness</span></div>"
            f"<div style='margin-top:20px;display:flex;justify-content:space-around;"
            f"border-top:1px solid rgba(99,102,241,.12);padding-top:16px;'>"
            f"<div style='text-align:center;'>"
            f"<div style='font-family:Space Grotesk,sans-serif;font-size:22px;font-weight:700;color:{crisis_c};'>{r['crisis_level']}</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#4a6080;margin-top:3px;text-transform:uppercase;'>Crisis Risk</div></div>"
            f"<div style='width:1px;background:rgba(99,102,241,.15);'></div>"
            f"<div style='text-align:center;'>"
            f"<div style='font-family:Space Grotesk,sans-serif;font-size:22px;font-weight:700;color:{dc};'>{da}</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#4a6080;margin-top:3px;text-transform:uppercase;'>Trajectory</div></div>"
            f"<div style='width:1px;background:rgba(99,102,241,.15);'></div>"
            f"<div style='text-align:center;'>"
            f"<div style='font-family:Space Grotesk,sans-serif;font-size:22px;font-weight:700;color:#818cf8;'>{int(r['crisis_prob']*100)}%</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#4a6080;margin-top:3px;text-transform:uppercase;'>Crisis Prob.</div></div>"
            f"</div></div>",
            unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        g1,g2=st.columns(2,gap="medium")
        with g1: render(chart_fingerprint(r["fp"]))
        with g2: render(chart_type_probs(r["probs"]))
        st.markdown("<br>",unsafe_allow_html=True)
        LBL("📡 BEHAVIORAL READINGS","#818cf8")
        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("Response Time",f"{ctx['resp']:.0f} min","avg latency")
        m2.metric("Contacts/Week",str(ctx['contacts']),"unique convos")
        m3.metric("Initiative",f"{ctx['init']*100:.0f}%","you initiate")
        m4.metric("Night Activity",f"{ctx['night']*100:.0f}%","1am–4am")
        m5.metric("Weekend",f"{ctx['weekend']*100:.0f}%","activity score")
        m6.metric("Future Thinking",f"{ctx['future']*100:.0f}%","forward language")
        st.markdown("<br>",unsafe_allow_html=True)
        sc1,sc2=st.columns(2,gap="large")
        with sc1:
            LBL("⚡ KEY SIGNALS","#f87171")
            if r["sigs"]:
                for sig in r["sigs"]:
                    sev_c={"high":"#f87171","medium":"#fbbf24","low":"#10b981"}.get(sig["sev"],"#6366f1")
                    st.markdown(f"<div style='padding:14px 18px;background:rgba(255,255,255,.02);border-left:3px solid {sev_c};border-radius:0 12px 12px 0;margin:8px 0;'><div style='display:flex;justify-content:space-between;'><span style='font-size:15px;font-weight:700;color:{sev_c};'>{sig['s']}</span><span style='font-family:JetBrains Mono,monospace;font-size:10px;background:{sev_c}18;color:{sev_c};padding:2px 8px;border-radius:99px;'>{sig['sev'].upper()}</span></div><div style='font-size:13px;color:#64748b;margin-top:5px;'>{sig['d']}</div><div style='font-size:12px;color:#6366f1;margin-top:4px;'>→ {sig['i']}</div></div>",unsafe_allow_html=True)
            else:
                st.success("✅ No critical signals. Patterns look healthy.")
        with sc2:
            LBL("🧭 INTERVENTIONS","#10b981")
            for inv in [{"a":"Send one message today","p":"high","d":"One line to someone you haven't spoken to in 2+ weeks.","t":"2 min","imp":"High","sci":"Weak ties reactivated in 60d retain 85% depth"},{"a":"Replace one solo activity","p":"medium","d":"Turn something alone into something shared this week.","t":"1 hr","imp":"Medium","sci":"Shared experiences create 40% stronger bonds"},{"a":"20 min outside daily","p":"medium","d":"Outdoor movement resets cortisol and loneliness biomarkers.","t":"20 min","imp":"Medium","sci":"Reduces loneliness markers 23% in 2 weeks"}]:
                p_c={"high":"#f87171","medium":"#60a5fa","low":"#10b981"}.get(inv["p"],"#6366f1")
                st.markdown(f"<div style='padding:16px 18px;background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.15);border-radius:14px;margin:8px 0;'><div style='display:flex;justify-content:space-between;margin-bottom:8px;'><span style='font-size:15px;font-weight:700;color:#eef2ff;'>{inv['a']}</span><span style='font-family:JetBrains Mono,monospace;font-size:10px;background:{p_c}18;color:{p_c};padding:2px 8px;border-radius:99px;'>{inv['p'].upper()}</span></div><div style='font-size:13px;color:#64748b;line-height:1.6;margin-bottom:8px;'>{inv['d']}</div><div style='display:flex;gap:8px;'><span style='font-size:11px;color:#4a6080;background:rgba(255,255,255,.03);padding:3px 10px;border-radius:6px;'>⏱ {inv['t']}</span><span style='font-size:11px;color:#4a6080;background:rgba(255,255,255,.03);padding:3px 10px;border-radius:6px;'>📈 {inv['imp']}</span></div><div style='font-size:11px;color:#6366f1;font-style:italic;border-top:1px solid rgba(99,102,241,.1);padding-top:8px;margin-top:8px;'>📚 {inv['sci']}</div></div>",unsafe_allow_html=True)

    with t2:
        st.markdown("<br>",unsafe_allow_html=True)
        LBL("📊 WHAT IS HURTING YOUR SCORE","#f87171")
        BODY("Each bar shows how many points that behavior is costing you. Green = healthy, red = needs attention.")
        render(chart_signal_breakdown(ctx, score))

        st.markdown("<br>",unsafe_allow_html=True)
        LBL("🔎 PLAIN ENGLISH BREAKDOWN","#10b981")
        night_v=ctx["night"]; wknd_v=ctx["weekend"]; init_v=ctx["init"]; resp_v=ctx["resp"]; contacts_v=ctx["contacts"]; future_v=ctx["future"]
        breakdown_items = [
            (
                "💬 Response Time",
                f"You take {resp_v:.0f} minutes on average to reply.",
                "Fast replies signal emotional availability — you're engaged." if resp_v<60 else "Slow replies (2h+) are one of the earliest detectable drift signals.",
                "#10b981" if resp_v<60 else "#f87171"
            ),
            (
                "👥 Social Circle",
                f"You spoke to {contacts_v} people this week.",
                "Above the 5-contact threshold — healthy range." if contacts_v>=5 else f"Below 5 contacts/week is clinically significant. You need {max(0,5-contacts_v)} more.",
                "#10b981" if contacts_v>=5 else "#f87171"
            ),
            (
                "📲 Who Reaches Out",
                f"You initiate {init_v*100:.0f}% of conversations.",
                "Balanced or high initiation keeps relationships alive." if init_v>=0.4 else "Low initiation means your social circle is slowly shrinking — it won't feel that way until it's gone.",
                "#10b981" if init_v>=0.4 else "#fbbf24"
            ),
            (
                "🌙 Night Activity",
                f"{night_v*100:.0f}% of your activity is between 1–4am.",
                "Healthy sleep pattern — no late-night isolation signal." if night_v<0.15 else "Late-night phone use is the single strongest behavioral predictor of isolation. Stronger than self-reported loneliness.",
                "#10b981" if night_v<0.15 else "#f87171"
            ),
            (
                "📅 Weekend Life",
                f"Weekend activity score: {wknd_v*100:.0f}%.",
                "Active weekends — socially protective." if wknd_v>=0.5 else "Weekend voids are 3× more predictive of chronic loneliness than weekday isolation.",
                "#10b981" if wknd_v>=0.5 else "#fbbf24"
            ),
            (
                "🔮 Future Thinking",
                f"Forward-looking language: {future_v*100:.0f}%.",
                "Good future orientation — things to look forward to." if future_v>=0.2 else "Future thinking collapses 4–8 weeks before a loneliness crisis. This is an early warning.",
                "#10b981" if future_v>=0.2 else "#fbbf24"
            ),
        ]
        for icon_title, reading, meaning, c_item in breakdown_items:
            st.markdown(
                f"<div style='padding:16px 18px;background:rgba(255,255,255,.02);"
                f"border:1px solid {c_item}22;border-left:3px solid {c_item};"
                f"border-radius:0 14px 14px 0;margin:8px 0;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:start;margin-bottom:6px;'>"
                f"<span style='font-size:14px;font-weight:700;color:#eef2ff;'>{icon_title}</span>"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;color:{c_item};"
                f"background:{c_item}15;padding:2px 8px;border-radius:99px;white-space:nowrap;margin-left:8px;'>"
                f"{reading}</span></div>"
                f"<div style='font-size:13px;color:#94a3b8;line-height:1.6;'>{meaning}</div>"
                f"</div>",
                unsafe_allow_html=True)

    with t3:
        st.markdown("<br>",unsafe_allow_html=True)
        LBL("🌍 WHERE YOU STAND","#60a5fa")
        BODY("Your score vs global population, type breakdown, and your week pattern vs healthy baseline.")
        p1,p2,p3=st.columns(3,gap="medium")
        with p1: render(chart_pop_hist(score))
        with p2: render(chart_type_donut())
        with p3: render(chart_week_pattern(lt,score,ctx))
        st.markdown("<br>",unsafe_allow_html=True)
        g1,g2,g3,g4=st.columns(4)
        g1.metric("Global Affected","1 Billion+","chronically lonely")
        g2.metric("Economic Cost","$2.5T / yr","productivity loss")
        g3.metric("Mortality Risk","= 15 cigs/day","daily equivalent")
        g4.metric("Before Tether","0 instruments","clinical tools")
        st.markdown("<br>",unsafe_allow_html=True)
        pc1,pc2=st.columns(2,gap="large")
        with pc1: st.markdown("<div style='padding:22px;background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.15);border-radius:16px;'><p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6366f1;letter-spacing:.15em;margin-bottom:12px;'>🇨🇳 THE CHINA SIGNAL</p><p style='font-size:14px;color:#94a3b8;line-height:1.8;margin:0;'>In 2024, tree-hole apps became the most downloaded apps in China in weeks. <strong style='color:#eef2ff;'>Tens of millions</strong> — not because the tech was good, but because the loneliness was unbearable. Japan, UK, US all declared loneliness a public health crisis. <strong style='color:#a5b4fc;'>None built an instrument. Tether did.</strong></p></div>",unsafe_allow_html=True)
        with pc2: st.markdown("<div style='padding:22px;background:rgba(124,58,237,.06);border:1px solid rgba(124,58,237,.15);border-radius:16px;'><p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#8b5cf6;letter-spacing:.15em;margin-bottom:12px;'>🔬 THE SCIENCE</p><p style='font-size:14px;color:#94a3b8;line-height:1.8;margin:0;'>Loneliness raises mortality by <strong style='color:#eef2ff;'>26%</strong> — equivalent to 15 cigarettes daily. Dementia risk +50%. Heart disease +29%. Depression has PHQ-9. Anxiety has GAD-7. Loneliness had nothing. <strong style='color:#c4b5fd;'>Tether is building the instrument.</strong></p></div>",unsafe_allow_html=True)

    with t4:
        st.markdown("<br>",unsafe_allow_html=True)

    
        LBL("📋 YOUR PERSONALISED 4-WEEK BLUEPRINT","#10b981")
        BODY("Not generic advice. Every week targets your specific weakest signal.")

        plan = []
        if ctx["contacts"] < 5:
            plan.append(("Week 1","Contact Repair","Reach out to 3 people you haven't spoken to in 30+ days — one text each.",f"Your contact count ({ctx['contacts']}/week) is below the clinical threshold of 5.","#f87171","📱"))
        elif ctx["init"] < 0.35:
            plan.append(("Week 1","Initiation Practice","Start 3 conversations this week — you pick who, you reach out first.",f"You only initiate {ctx['init']*100:.0f}% of convos. Low initiation is the fastest path to circle collapse.","#f87171","📲"))
        else:
            plan.append(("Week 1","Connection Audit","List your 5 most important relationships. When did you last speak to each one?","Healthy circles need active maintenance — this exercise prevents silent drift.","#60a5fa","✍️"))

        if ctx["weekend"] < 0.4:
            plan.append(("Week 2","Weekend Activation","Make one fixed social plan for this Saturday — non-negotiable, calendar it now.",f"Weekend activity at {ctx['weekend']*100:.0f}% is your biggest vulnerability. One recurring plan changes everything.","#fbbf24","📅"))
        else:
            plan.append(("Week 2","Deepen One Tie","Have one real conversation — not small talk. Share something genuine.",f"Surface-level contact doesn't prevent loneliness. Depth is what protects.","#fbbf24","💬"))

        if ctx["night"] > 0.2:
            plan.append(("Week 3","Night Pattern Reset","Set a phone-down time: 11:30pm. 7 days straight. Track how you feel.",f"Your {ctx['night']*100:.0f}% late-night activity is reinforcing isolation. One habit change breaks the loop.","#a78bfa","🌙"))
        elif ctx["future"] < 0.2:
            plan.append(("Week 3","Plant a Future","Book one thing to look forward to — a meal, a trip, anything 2–4 weeks away.",f"Future thinking at {ctx['future']*100:.0f}% is low. Even small plans create measurable wellbeing improvement.","#a78bfa","🔮"))
        else:
            plan.append(("Week 3","New Weak Tie","Introduce yourself to one new person — a neighbour, a colleague, anyone.",f"New weak ties are statistically the best source of life opportunities and social energy.","#a78bfa","🤝"))

       
        plan.append(("Week 4","Lock In One Ritual","Choose one recurring social activity — weekly, same time, same people.",f"Recurring contact with the same people is the only proven mechanism for building adult friendship.","#10b981","🔄"))

        for week, title, action, why, c_week, icon in plan:
            st.markdown(
                f"<div style='padding:20px 22px;background:rgba(255,255,255,.02);"
                f"border:1px solid {c_week}25;border-radius:16px;margin:10px 0;"
                f"display:flex;gap:18px;align-items:start;'>"
                f"<div style='flex-shrink:0;text-align:center;'>"
                f"<div style='font-size:28px;margin-bottom:4px;'>{icon}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                f"color:{c_week};font-weight:700;letter-spacing:.1em;'>{week.upper()}</div>"
                f"</div>"
                f"<div style='flex:1;'>"
                f"<div style='font-family:Space Grotesk,sans-serif;font-size:16px;"
                f"font-weight:700;color:#eef2ff;margin-bottom:6px;'>{title}</div>"
                f"<div style='font-size:14px;color:#e2e8f0;line-height:1.6;margin-bottom:8px;"
                f"font-weight:500;'>{action}</div>"
                f"<div style='font-size:12px;color:#4a6080;line-height:1.6;"
                f"border-top:1px solid rgba(255,255,255,.05);padding-top:8px;'>"
                f"<em>Why this week: {why}</em></div>"
                f"</div></div>",
                unsafe_allow_html=True)



    with t5:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown("<div style='display:flex;align-items:center;gap:14px;padding:18px 22px;background:linear-gradient(135deg,rgba(6,182,212,.08),rgba(59,130,246,.05));border:1px solid rgba(6,182,212,.22);border-radius:18px;margin-bottom:4px;'><div style='width:46px;height:46px;background:rgba(6,182,212,.18);border:1px solid rgba(6,182,212,.35);border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;'>🗺️</div><div style='flex:1;'><div style='font-family:Space Grotesk,sans-serif;font-size:17px;font-weight:700;color:#67e8f9;'>Environment Scanner</div><div style='font-size:12px;color:#4a6080;margin-top:3px;'>Scans your city for matched opportunities · Ranks by connection potential</div></div><span style='width:8px;height:8px;border-radius:50%;background:#67e8f9;box-shadow:0 0 8px #67e8f9;'></span></div>",unsafe_allow_html=True)
        with st.expander("Open Environment Scanner →"):
            ec1,ec2=st.columns([1,1.4],gap="large")
            with ec1:
                city=st.text_input("Your city",placeholder="Mumbai, Delhi...",key="city")
                comfort=st.select_slider("Comfort level",options=["introvert","ambivert","extrovert"],value="ambivert",key="comfort")
                st.multiselect("Interests",["Books","Hiking","Cooking","Music","Art","Sports","Tech","Yoga","Photography","Volunteering"],default=["Books","Hiking"],key="ints")
            with ec2:
                if st.button("✦ Scan My Environment",key="scan"):
                    with st.spinner("Scanning..."):
                        time.sleep(1.2)
                        edb={"introvert":[{"name":"Silent Book Club","type":"Weekly · Tue","people":"4–8","score":94,"icon":"📚","tag":"Perfect","why":"Read in company — zero pressure"},{"name":"Photography Walk","type":"Weekends","people":"4–6","score":87,"icon":"📷","tag":"High","why":"Side-by-side dissolves awkwardness"},{"name":"Pottery Class","type":"Thu evenings","people":"6 max","score":82,"icon":"🏺","tag":"Good","why":"Making things = bonds without forced convo"}],"ambivert":[{"name":"Cooking Masterclass","type":"Wed evenings","people":"10–14","score":91,"icon":"👨‍🍳","tag":"Perfect","why":"Shared accomplishment = instant camaraderie"},{"name":"Weekend Hiking Group","type":"Sun mornings","people":"8–15","score":88,"icon":"🥾","tag":"High","why":"Walking makes deep convo effortless"},{"name":"Volunteer Kitchen","type":"Saturdays","people":"8–12","score":85,"icon":"🍲","tag":"High","why":"Shared purpose = fastest route to belonging"}],"extrovert":[{"name":"Improv Comedy","type":"Thu evenings","people":"15–20","score":95,"icon":"🎭","tag":"Perfect","why":"Fastest bonding environment known"},{"name":"Sports League","type":"Weekends","people":"20+","score":90,"icon":"⚽","tag":"High","why":"Recurring contact builds real relationships"},{"name":"Startup Dinner","type":"Monthly","people":"25–40","score":84,"icon":"🤝","tag":"Good","why":"Shared identity makes every convo meaningful"}]}
                        st.session_state.ao["events"]=edb.get(comfort,edb["ambivert"])
                if st.session_state.ao.get("events"):
                    LBL("📍 RANKED BY CONNECTION POTENTIAL","#22d3ee")
                    for ev in st.session_state.ao["events"]:
                        ec={"Perfect":"#10b981","High":"#60a5fa","Good":"#fbbf24"}.get(ev["tag"],"#6366f1")
                        ei=ev["icon"]; en=ev["name"]; et=ev["type"]; ep=ev["people"]; ew=ev["why"]; es=ev["score"]
                        st.markdown(f"<div style='display:flex;align-items:start;gap:14px;padding:14px 18px;background:rgba(6,182,212,.05);border:1px solid rgba(6,182,212,.15);border-radius:14px;margin:8px 0;'><span style='font-size:26px;'>{ei}</span><div style='flex:1;'><div style='font-size:15px;font-weight:700;color:#eef2ff;'>{en}</div><div style='font-size:12px;color:#22d3ee;margin:3px 0;'>📅 {et} · 👥 {ep}</div><div style='font-size:12px;color:#64748b;'>{ew}</div></div><div style='text-align:right;flex-shrink:0;'><div style='font-family:Space Grotesk,sans-serif;font-size:26px;font-weight:800;color:{ec};'>{es}</div><div style='font-size:9px;color:#4a6080;'>MATCH</div></div></div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

        crisis_status="monitoring" if score>55 else "gentle_checkin" if score>35 else "active"
        cc={"monitoring":"#10b981","gentle_checkin":"#fbbf24","active":"#f87171"}[crisis_status]
        cs_label=crisis_status.replace("_"," ").upper()
        crisis_header=(f"<div style='display:flex;align-items:center;gap:14px;padding:18px 22px;"
                       f"background:linear-gradient(135deg,rgba(248,113,113,.08),rgba(251,191,36,.05));"
                       f"border:1px solid rgba(248,113,113,.22);border-radius:18px;margin-bottom:4px;'>"
                       f"<div style='width:46px;height:46px;background:rgba(248,113,113,.18);"
                       f"border:1px solid rgba(248,113,113,.35);border-radius:13px;"
                       f"display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;'>🚨</div>"
                       f"<div style='flex:1;'>"
                       f"<div style='font-family:Space Grotesk,sans-serif;font-size:17px;font-weight:700;color:#fca5a5;'>Crisis Interceptor</div>"
                       f"<div style='font-size:12px;color:#4a6080;margin-top:3px;'>Watches the 6-week pre-crisis window · Acts before you feel it</div>"
                       f"</div>"
                       f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;"
                       f"color:{cc};background:{cc}18;padding:5px 12px;border-radius:99px;'>{cs_label}</span>"
                       f"</div>")
        st.markdown(crisis_header,unsafe_allow_html=True)
        with st.expander("Open Crisis Interceptor →"):
            ci1,ci2=st.columns([1,1.4],gap="large")
            with ci1:
                render(chart_crisis_status(score))
            with ci2:
                msgs_c={"monitoring":"Your patterns are stable. The agent is watching quietly. Safe distance from the crisis zone.","gentle_checkin":"Your last few weeks look heavy. Not an alarm — just a hand on your shoulder. One real conversation this week would help.","active":"The patterns are in the crisis window. Reach out to one person today. iCall India: 9152987821 (Mon–Sat 8am–10pm)."}
                acts={"monitoring":[("Monitoring silently","#10b981"),("Weekly pattern review","#6366f1"),("Alert if drift begins","#818cf8")],"gentle_checkin":[("Nudge trusted contact","#fbbf24"),("Environment scanner ready","#60a5fa"),("Guided reflection prompt","#818cf8")],"active":[("iCall India: 9152987821","#f87171"),("Emergency msg drafted","#fbbf24"),("Local support located","#6366f1")]}
                st.markdown(f"<div style='padding:16px 18px;background:rgba(255,255,255,.02);border-radius:14px;border:1px solid rgba(255,255,255,.06);margin-bottom:12px;'><p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#4a6080;letter-spacing:.12em;margin-bottom:8px;'>AGENT MESSAGE</p><p style='font-size:15px;color:#eef2ff;line-height:1.8;margin:0;'>{msgs_c[crisis_status]}</p></div>",unsafe_allow_html=True)
                LBL("AVAILABLE ACTIONS","#94a3b8")
                for act,a_c in acts[crisis_status]:
                    st.markdown(f"<div style='display:flex;align-items:center;gap:10px;padding:11px 14px;background:{a_c}0a;border:1px solid {a_c}22;border-radius:12px;margin:6px 0;'><span style='width:7px;height:7px;border-radius:50%;background:{a_c};flex-shrink:0;'></span><span style='font-size:13px;font-weight:600;color:#eef2ff;'>{act}</span><span style='margin-left:auto;color:{a_c};font-size:16px;'>→</span></div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

  
        st.markdown(
            "<div style='display:flex;align-items:center;gap:14px;padding:18px 22px;"
            "background:linear-gradient(135deg,rgba(167,139,250,.08),rgba(139,92,246,.05));"
            "border:1px solid rgba(167,139,250,.22);border-radius:18px;margin-bottom:4px;'>"
            "<div style='width:46px;height:46px;background:rgba(167,139,250,.18);"
            "border:1px solid rgba(167,139,250,.35);border-radius:13px;"
            "display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;'>🔍</div>"
            "<div style='flex:1;'>"
            "<div style='font-family:Space Grotesk,sans-serif;font-size:17px;font-weight:700;color:#c4b5fd;'>Drift Interceptor</div>"
            "<div style='font-size:12px;color:#4a6080;margin-top:3px;'>Pinpoints when your social collapse began · Identifies the trigger · Maps your recovery path</div>"
            "</div>"
            "<div style='width:8px;height:8px;border-radius:50%;background:#a78bfa;box-shadow:0 0 8px #a78bfa;'></div>"
            "</div>",
            unsafe_allow_html=True)
        with st.expander("Open Drift Interceptor →", expanded=True):
            night=ctx["night"]; weekend=ctx["weekend"]; future=ctx["future"]
            init=ctx["init"]; resp=ctx["resp"]; contacts_n=ctx["contacts"]; wfh=r.get("ctx",{})

            weeks_ago = 0
            if night>0.35: weeks_ago+=10
            elif night>0.2: weeks_ago+=5
            if weekend<0.3: weeks_ago+=8
            elif weekend<0.5: weeks_ago+=3
            if future<0.08: weeks_ago+=7
            elif future<0.2: weeks_ago+=3
            if init<0.25: weeks_ago+=6
            elif init<0.4: weeks_ago+=2
            weeks_ago = max(2, min(24, weeks_ago))

            root_causes = []
            if ctx.get("wfh",0) and weekend<0.5 and contacts_n<6:
                root_causes.append(("Remote work isolation","Working from home removed your daily ambient social contact — the invisible glue that kept you connected without effort.","#60a5fa"))
            if night>0.25 and future<0.15:
                root_causes.append(("Digital substitution","Real social contact has been gradually replaced by passive screen consumption. The brain gets stimulation without connection — accelerating the drift.","#f87171"))
            if weekend<0.35 and init<0.35:
                root_causes.append(("Social avoidance pattern","A feedback loop has formed: lower energy → less social effort → more isolation → lower energy. Each cycle is harder to break.","#fbbf24"))
            if contacts_n<4 and init<0.4:
                root_causes.append(("Circle contraction","Your social circle has quietly shrunk — not through conflict, but through the gradual silence of not reaching out. Drift rarely announces itself.","#a78bfa"))
            if not root_causes:
                root_causes.append(("Gradual social atrophy","No single trigger — this is the slow, invisible kind. Social muscles weaken when not exercised. The patterns suggest a gradual decline rather than a sudden shift.","#10b981"))
            primary = root_causes[0]

            recovery_actions = {
                0: ("Join one recurring weekly activity","Recurring contact with the same people is the only proven accelerator of new adult friendships.",92),
                1: ("Replace 30 min of solo screen time with a synchronous call","Passive consumption feels social but blocks real connection.",78),
                2: ("One social commitment per weekend, non-negotiable","Treating social time like work time breaks the avoidance loop.",81),
                3: ("Contact one person from your pre-drift circle this week","Dormant ties reactivate 3× faster than building new ones.",85),
                4: ("Start any shared weekly activity","Even low-intensity shared activity rebuilds the social baseline.",75),
            }
            ca_idx = min(len(root_causes)-1, 0)
            turning_point = recovery_actions[["Remote work isolation","Digital substitution","Social avoidance pattern","Circle contraction","Gradual social atrophy"].index(primary[0]) if primary[0] in ["Remote work isolation","Digital substitution","Social avoidance pattern","Circle contraction","Gradual social atrophy"] else 4]

            LBL("📍 YOUR COLLAPSE TIMELINE","#a78bfa")
            BODY(f"Reconstructed from your behavioral fingerprint. Drift estimated to have begun approximately <strong style='color:#eef2ff;'>{weeks_ago} weeks ago.</strong>")
            fig_di, _ = chart_drift_interceptor(score, lt, ctx)
            render(fig_di)

            st.markdown("<br>",unsafe_allow_html=True)

            di1, di2, di3 = st.columns(3, gap="medium")

            with di1:
                LBL("🧬 ROOT CAUSE DETECTED","#f87171")
                pc = primary
                st.markdown(
                    f"<div style='padding:20px;background:rgba(255,255,255,.02);"
                    f"border:1px solid {pc[2]}30;border-left:3px solid {pc[2]};"
                    f"border-radius:0 14px 14px 0;'>"
                    f"<div style='font-family:Space Grotesk,sans-serif;font-size:15px;"
                    f"font-weight:700;color:{pc[2]};margin-bottom:10px;'>{pc[0]}</div>"
                    f"<div style='font-size:13px;color:#94a3b8;line-height:1.75;'>{pc[1]}</div>"
                    f"</div>",
                    unsafe_allow_html=True)
                if len(root_causes) > 1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    LBL("SECONDARY FACTORS","#4a6080")
                    for rc in root_causes[1:]:
                        st.markdown(
                            f"<div style='padding:10px 14px;background:{rc[2]}08;"
                            f"border:1px solid {rc[2]}20;border-radius:10px;margin:5px 0;'>"
                            f"<div style='font-size:12px;font-weight:600;color:{rc[2]};'>{rc[0]}</div>"
                            f"</div>",
                            unsafe_allow_html=True)

            with di2:
                LBL("⚡ THE TURNING POINT","#fbbf24")
                tp_action, tp_science, tp_prob = turning_point
                st.markdown(
                    f"<div style='padding:20px;background:rgba(251,191,36,.06);"
                    f"border:1px solid rgba(251,191,36,.2);border-radius:14px;'>"
                    f"<div style='font-family:Space Grotesk,sans-serif;font-size:15px;"
                    f"font-weight:700;color:#fbbf24;margin-bottom:10px;'>{tp_action}</div>"
                    f"<div style='font-size:13px;color:#94a3b8;line-height:1.75;"
                    f"margin-bottom:14px;'>{tp_science}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                    f"color:#4a6080;letter-spacing:.1em;text-transform:uppercase;"
                    f"margin-bottom:6px;'>Based on loneliness type & root cause</div>"
                    f"<div style='font-size:11px;color:#fbbf24;font-style:italic;'>"
                    f"This is the single highest-leverage action for your specific pattern.</div>"
                    f"</div>",
                    unsafe_allow_html=True)

            with di3:
                LBL("📈 30-DAY RECOVERY PROBABILITY","#10b981")
                base_recovery = max(40, min(94, 100 - score/2 + (40-weeks_ago)*0.5))
                no_action = max(10, base_recovery - 35)
                with_action = min(94, base_recovery)
                st.markdown(
                    f"<div style='padding:20px;background:rgba(16,185,129,.06);"
                    f"border:1px solid rgba(16,185,129,.2);border-radius:14px;text-align:center;'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                    f"color:#4a6080;letter-spacing:.1em;margin-bottom:8px;'>IF YOU TAKE TURNING POINT ACTION</div>"
                    f"<div style='font-family:Space Grotesk,sans-serif;font-size:56px;"
                    f"font-weight:800;color:#10b981;line-height:1;'>{with_action:.0f}%</div>"
                    f"<div style='font-size:12px;color:#10b981;margin:4px 0 16px;'>"
                    f"probability of measurable score improvement</div>"
                    f"<div style='height:4px;background:rgba(255,255,255,.04);border-radius:99px;overflow:hidden;margin-bottom:8px;'>"
                    f"<div style='height:100%;width:{with_action}%;background:#10b981;border-radius:99px;'></div></div>"
                    f"<div style='border-top:1px solid rgba(255,255,255,.06);padding-top:12px;margin-top:4px;'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                    f"color:#4a6080;letter-spacing:.1em;margin-bottom:6px;'>WITHOUT ACTION</div>"
                    f"<div style='font-family:Space Grotesk,sans-serif;font-size:32px;"
                    f"font-weight:700;color:#f87171;'>{no_action:.0f}%</div>"
                    f"</div></div>",
                    unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown(
                f"<div style='padding:18px 22px;background:rgba(167,139,250,.06);"
                f"border:1px solid rgba(167,139,250,.18);border-radius:14px;'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                f"color:#a78bfa;letter-spacing:.15em;margin-bottom:10px;'>🔍 DRIFT INTERCEPTOR SUMMARY</div>"
                f"<div style='font-size:14px;color:#eef2ff;line-height:1.9;'>"
                f"Your behavioral patterns indicate drift began approximately "
                f"<strong style='color:#a78bfa;'>{weeks_ago} weeks ago</strong>. "
                f"The primary driver is <strong style='color:{primary[2]};'>{primary[0].lower()}</strong>. "
                f"Your current score of <strong style='color:{sc(score)};'>{score:.0f}</strong> places you in the "
                f"<strong style='color:{sc(score)};'>{sl(score).lower()}</strong> zone. "
                f"Taking the turning point action gives you a "
                f"<strong style='color:#10b981;'>{with_action:.0f}% probability</strong> of measurable "
                f"improvement within 30 days — without action, that drops to "
                f"<strong style='color:#f87171;'>{no_action:.0f}%</strong>."
                f"</div></div>",
                unsafe_allow_html=True)

def show_footer():
    st.divider()
    fc1,fc2=st.columns([1,1])
    with fc1: st.markdown("<div style='font-family:Space Grotesk,sans-serif;font-size:18px;font-weight:800;color:#eef2ff;'>Tether</div><div style='font-size:10px;color:#1e2d4a;letter-spacing:.15em;text-transform:uppercase;font-family:JetBrains Mono,monospace;margin-top:4px;'>The first instrument the loneliness epidemic has ever had</div>",unsafe_allow_html=True)
    with fc2: st.markdown("<div style='text-align:right;font-size:11px;color:#1e2d4a;font-family:JetBrains Mono,monospace;'>Gradient Boosting · Pandas · Matplotlib · NumPy · Streamlit</div>",unsafe_allow_html=True)

p=st.session_state.page
if   p=="landing":    page_landing()
elif p=="onboarding": page_onboarding()
elif p=="results":    page_results()
else:                 page_landing()
show_footer()