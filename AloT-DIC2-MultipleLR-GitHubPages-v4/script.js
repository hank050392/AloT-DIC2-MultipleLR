const DATASETS = {
  startup: {
    url: "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv",
    target: "Profit"
  },
  boston: {
    url: "https://raw.githubusercontent.com/hotcode002/Machine-Learning-with-Python/master/week_2_math/data/boston_housing.csv",
    target: "medv"
  }
};

let rows = [];
let targetColumn = "";
let expandedRows = [];
let expandedFeatures = [];
let progression = [];
let selectedK = 1;
let currentModel = null;
let currentMethod = "lasso";
let currentSplit = null;
let rmseChart = null;
let r2Chart = null;

const METHOD_INFO = {
  lasso: "Lasso：先標準化訓練資料，以 L1 regularization 的係數大小進行 Feature Ranking，再用選出的 Feature 重新配適 OLS。",
  sfs: "Forward SFS：從空集合開始，每一步加入一個最能降低 training RMSE 的 Feature，直到達到指定 k。",
  rfe: "RFE：從全部 Feature 開始，反覆移除標準化 OLS 中絕對係數最小的 Feature，直到剩下指定 k。",
  elastic: "ElasticNet：先標準化訓練資料，以 L1 + L2 regularization 係數大小進行 Feature Ranking，再重新配適 OLS。",
  exhaustive: "Exhaustive Search：列舉所有 k 個 Feature 的組合，以 training RMSE 選出最佳子集合，再於 test set 評估。"
};

const $ = id => document.getElementById(id);

function mean(a){ return a.reduce((s,v)=>s+v,0)/a.length; }
function std(a){
  const m=mean(a);
  return Math.sqrt(mean(a.map(v=>(v-m)**2))) || 1;
}
function rmse(y,p){ return Math.sqrt(mean(y.map((v,i)=>(v-p[i])**2))); }
function mae(y,p){ return mean(y.map((v,i)=>Math.abs(v-p[i]))); }
function r2(y,p){
  const m=mean(y), ssTot=y.reduce((s,v)=>s+(v-m)**2,0), ssRes=y.reduce((s,v,i)=>s+(v-p[i])**2,0);
  return ssTot === 0 ? 0 : 1-ssRes/ssTot;
}
function adjustedR2(r2v,n,p){ return n-p-1>0 ? 1-(1-r2v)*(n-1)/(n-p-1) : NaN; }

function combinations(arr,k){
  const out=[];
  function rec(start,cur){
    if(cur.length===k){out.push(cur.slice());return;}
    for(let i=start;i<arr.length;i++){cur.push(arr[i]);rec(i+1,cur);cur.pop();}
  }
  rec(0,[]);
  return out;
}

function seededRandom(seed){
  let t=seed+0x6D2B79F5;
  return function(){
    t+=0x6D2B79F5;
    let r=Math.imul(t^t>>>15,1|t);
    r^=r+Math.imul(r^r>>>7,61|r);
    return ((r^r>>>14)>>>0)/4294967296;
  }
}

function splitData(data, ratio=.2){
  const rand=seededRandom(42);
  const idx=data.map((_,i)=>i);
  for(let i=idx.length-1;i>0;i--){
    const j=Math.floor(rand()*(i+1));
    [idx[i],idx[j]]=[idx[j],idx[i]];
  }
  const testN=Math.max(1,Math.round(data.length*ratio));
  const testSet=new Set(idx.slice(0,testN));
  return {
    train:data.filter((_,i)=>!testSet.has(i)),
    test:data.filter((_,i)=>testSet.has(i))
  };
}

function fitOLS(data,features,target){
  const X=data.map(r=>[1,...features.map(f=>Number(r[f]))]);
  const y=data.map(r=>Number(r[target]));
  const Xm=math.matrix(X), ym=math.matrix(y.map(v=>[v]));
  try{
    const Xt=math.transpose(Xm);
    const gram=math.multiply(Xt,Xm);
    const beta=math.multiply(math.inv(gram),math.multiply(Xt,ym))
      .toArray().map(v=>v[0]);
    return {features,beta};
  }catch(e){
    try{
      // Tiny ridge fallback for nearly singular matrices
      const Xt=math.transpose(Xm);
      let gram=math.multiply(Xt,Xm).toArray();
      for(let i=0;i<gram.length;i++) gram[i][i]+=1e-8;
      const beta=math.multiply(
        math.inv(math.matrix(gram)),
        math.multiply(Xt,ym)
      ).toArray().map(v=>v[0]);
      return {features,beta};
    }catch(_){ return null; }
  }
}

function predict(model,data){
  return data.map(r =>
    model.beta[0] +
    model.features.reduce((s,f,i)=>s+model.beta[i+1]*Number(r[f]),0)
  );
}

function evaluateModel(model, data){
  if(!model || !data.length) return null;
  const pred=predict(model,data);
  const y=data.map(r=>Number(r[targetColumn]));
  const out={
    rmse:rmse(y,pred),
    mae:mae(y,pred),
    r2:r2(y,pred),
    pred
  };
  out.adjR2=adjustedR2(out.r2,data.length,model.features.length);
  return out;
}

// One-hot encode categorical columns and drop first category as reference
function prepareExpandedData(raw,target){
  const cols=Object.keys(raw[0]||{}).filter(c=>c!==target);
  const numericCols=[], categoricalCols=[];

  for(const c of cols){
    const vals=raw.map(r=>r[c]).filter(v=>v!==null&&v!=="");
    const numeric=vals.length>0 && vals.every(v=>Number.isFinite(Number(v)));
    (numeric?numericCols:categoricalCols).push(c);
  }

  const categoryMap={};
  for(const c of categoricalCols){
    categoryMap[c]=[...new Set(
      raw.map(r=>String(r[c]??"").trim()).filter(Boolean)
    )].sort();
  }

  const featureNames=[...numericCols];
  for(const c of categoricalCols){
    const cats=categoryMap[c];
    for(let i=1;i<cats.length;i++){
      featureNames.push(`${c}_${cats[i].replace(/\s+/g,"_")}`);
    }
  }

  const data=raw.map(r=>{
    const o={};
    for(const c of numericCols) o[c]=Number(r[c]);

    for(const c of categoricalCols){
      const cats=categoryMap[c];
      const value=String(r[c]??"").trim();
      for(let i=1;i<cats.length;i++){
        o[`${c}_${cats[i].replace(/\s+/g,"_")}`]=value===cats[i]?1:0;
      }
    }
    o[target]=Number(r[target]);
    return o;
  }).filter(r=>
    Number.isFinite(r[target]) &&
    featureNames.every(f=>Number.isFinite(r[f]))
  );

  return {data,featureNames,numericCols,categoricalCols,categoryMap};
}

function standardizeTraining(train, features){
  const stats={};
  for(const f of features){
    const vals=train.map(r=>Number(r[f]));
    stats[f]={mean:mean(vals),std:std(vals)};
  }
  const yvals=train.map(r=>Number(r[targetColumn]));
  const ystat={mean:mean(yvals),std:std(yvals)};
  const X=train.map(r=>features.map(f=>(Number(r[f])-stats[f].mean)/stats[f].std));
  const y=yvals.map(v=>(v-ystat.mean)/ystat.std);
  return {X,y,stats,ystat};
}

function softThreshold(z,gamma){
  if(z>gamma) return z-gamma;
  if(z<-gamma) return z+gamma;
  return 0;
}

function elasticNetCoefficients(train, features, alpha=0.05, l1Ratio=1.0, maxIter=1500){
  const {X,y}=standardizeTraining(train,features);
  const n=X.length, p=features.length;
  const beta=new Array(p).fill(0);

  for(let iter=0;iter<maxIter;iter++){
    let maxChange=0;
    for(let j=0;j<p;j++){
      let rho=0, z=0;
      for(let i=0;i<n;i++){
        let pred=0;
        for(let k=0;k<p;k++) if(k!==j) pred+=X[i][k]*beta[k];
        const residual=y[i]-pred;
        rho+=X[i][j]*residual;
        z+=X[i][j]*X[i][j];
      }
      rho/=n; z/=n;
      const l1=alpha*l1Ratio;
      const l2=alpha*(1-l1Ratio);
      const newB=softThreshold(rho,l1)/(z+l2);
      maxChange=Math.max(maxChange,Math.abs(newB-beta[j]));
      beta[j]=newB;
    }
    if(maxChange<1e-7) break;
  }
  return beta;
}

function correlationAbs(train, f){
  const x=train.map(r=>Number(r[f]));
  const y=train.map(r=>Number(r[targetColumn]));
  const mx=mean(x), my=mean(y);
  let num=0,dx=0,dy=0;
  for(let i=0;i<x.length;i++){
    const a=x[i]-mx,b=y[i]-my;
    num+=a*b; dx+=a*a; dy+=b*b;
  }
  return Math.abs(num/Math.sqrt((dx||1)*(dy||1)));
}

function topRankedFeatures(train, features, coeffs, k){
  let ranked=features.map((f,i)=>({f,score:Math.abs(coeffs[i])}))
    .sort((a,b)=>b.score-a.score);

  // If regularization zeroed too many coefficients, use correlation as tie-break/fallback.
  ranked=ranked.map(x=>({
    ...x,
    score:x.score>1e-10 ? x.score : correlationAbs(train,x.f)*1e-6
  })).sort((a,b)=>b.score-a.score);

  return ranked.slice(0,k).map(x=>x.f);
}

function selectLasso(train, features, k){
  const coef=elasticNetCoefficients(train,features,0.035,1.0);
  return topRankedFeatures(train,features,coef,k);
}

function selectElasticNet(train, features, k){
  const coef=elasticNetCoefficients(train,features,0.035,0.5);
  return topRankedFeatures(train,features,coef,k);
}

function selectSFS(train, features, k){
  const chosen=[];
  const remaining=[...features];

  while(chosen.length<k && remaining.length){
    let bestF=null,bestScore=Infinity;
    for(const f of remaining){
      const cand=[...chosen,f];
      const model=fitOLS(train,cand,targetColumn);
      if(!model) continue;
      const ev=evaluateModel(model,train);
      if(ev.rmse<bestScore){bestScore=ev.rmse;bestF=f;}
    }
    if(bestF===null) break;
    chosen.push(bestF);
    remaining.splice(remaining.indexOf(bestF),1);
  }
  return chosen;
}

function standardizedOlsCoefficients(train, features){
  const {X,y}=standardizeTraining(train,features);
  const pseudo=X.map((row,i)=>{
    const o={__y:y[i]};
    features.forEach((f,j)=>o[f]=row[j]);
    return o;
  });
  const oldTarget=targetColumn;
  targetColumn="__y";
  const model=fitOLS(pseudo,features,"__y");
  targetColumn=oldTarget;
  return model ? model.beta.slice(1) : new Array(features.length).fill(0);
}

function selectRFE(train, features, k){
  let current=[...features];
  while(current.length>k){
    const coefs=standardizedOlsCoefficients(train,current);
    let removeIdx=0,minAbs=Infinity;
    coefs.forEach((b,i)=>{
      const a=Math.abs(b);
      if(a<minAbs){minAbs=a;removeIdx=i;}
    });
    current.splice(removeIdx,1);
  }
  return current;
}

function selectExhaustive(train, features, k){
  let best=null,bestRmse=Infinity;
  for(const combo of combinations(features,k)){
    const model=fitOLS(train,combo,targetColumn);
    if(!model) continue;
    const ev=evaluateModel(model,train);
    if(ev.rmse<bestRmse){bestRmse=ev.rmse;best=combo;}
  }
  return best || features.slice(0,k);
}

function chooseFeatures(method, train, features, k){
  if(method==="lasso") return selectLasso(train,features,k);
  if(method==="sfs") return selectSFS(train,features,k);
  if(method==="rfe") return selectRFE(train,features,k);
  if(method==="elastic") return selectElasticNet(train,features,k);
  return selectExhaustive(train,features,k);
}

function buildProgression(method, train, test, features, maxK){
  const out=[];
  for(let k=1;k<=Math.min(maxK,features.length);k++){
    const selected=chooseFeatures(method,train,features,k);
    if(!selected || selected.length!==k) continue;
    const model=fitOLS(train,selected,targetColumn);
    if(!model) continue;
    const testEval=evaluateModel(model,test);
    out.push({
      k,features:selected,model,
      rmse:testEval.rmse,
      mae:testEval.mae,
      r2:testEval.r2,
      adjR2:testEval.adjR2
    });
  }
  return out;
}


function matrixInverseSafe(A){
  try{
    return math.inv(math.matrix(A)).toArray();
  }catch(e){
    const B=A.map((row,i)=>row.map((v,j)=>v+(i===j?1e-8:0)));
    return math.inv(math.matrix(B)).toArray();
  }
}

function computeVIF(data, selectedFeatures){
  const out={};
  for(const f of selectedFeatures){
    const others=selectedFeatures.filter(x=>x!==f);
    if(others.length===0){
      out[f]=1;
      continue;
    }
    const model=fitOLS(data,others,f);
    if(!model){
      out[f]=Infinity;
      continue;
    }
    const pred=predict(model,data);
    const actual=data.map(r=>Number(r[f]));
    const r2v=r2(actual,pred);
    out[f]=1/Math.max(1-r2v,1e-10);
  }
  return out;
}

function computeOlsStatistics(data, model, target){
  const features=model.features;
  const n=data.length;
  const p=features.length;
  const X=data.map(r=>[1,...features.map(f=>Number(r[f]))]);
  const y=data.map(r=>Number(r[target]));
  const pred=predict(model,data);

  const residuals=y.map((v,i)=>v-pred[i]);
  const sse=residuals.reduce((s,v)=>s+v*v,0);
  const yMean=mean(y);
  const sst=y.reduce((s,v)=>s+(v-yMean)**2,0);
  const dfResidual=n-p-1;
  const sigma2=dfResidual>0 ? sse/dfResidual : NaN;

  const Xt=math.transpose(math.matrix(X));
  const XtX=math.multiply(Xt,math.matrix(X)).toArray();
  const invXtX=matrixInverseSafe(XtX);

  const se=model.beta.map((_,i)=>{
    const v=sigma2*invXtX[i][i];
    return Math.sqrt(Math.max(v,0));
  });

  const tStats=model.beta.map((b,i)=>se[i]===0 ? NaN : b/se[i]);
  const pValues=tStats.map(t=>{
    if(!Number.isFinite(t) || dfResidual<=0 || typeof jStat==="undefined") return NaN;
    return 2*(1-jStat.studentt.cdf(Math.abs(t),dfResidual));
  });

  let tCrit=1.96;
  if(dfResidual>0 && typeof jStat!=="undefined"){
    tCrit=jStat.studentt.inv(0.975,dfResidual);
  }

  const ci=model.beta.map((b,i)=>[
    b-tCrit*se[i],
    b+tCrit*se[i]
  ]);

  let fStat=NaN, fP=NaN;
  if(p>0 && dfResidual>0 && sse>0){
    fStat=((sst-sse)/p)/(sse/dfResidual);
    if(typeof jStat!=="undefined"){
      fP=1-jStat.centralF.cdf(fStat,p,dfResidual);
    }
  }

  const vif=computeVIF(data,features);

  return {
    n,p,dfResidual,sse,sst,se,tStats,pValues,ci,fStat,fP,vif
  };
}

function formatP(p){
  if(!Number.isFinite(p)) return "-";
  if(p<0.0001) return "&lt; 0.0001";
  return p.toFixed(4);
}

function renderStatistics(model){
  if(!currentSplit || !model) return;

  // Statistical inference is computed on the training split.
  const stats=computeOlsStatistics(currentSplit.train,model,targetColumn);

  $("fStatistic").textContent=Number.isFinite(stats.fStat)?stats.fStat.toFixed(2):"-";
  $("fPValue").innerHTML=Number.isFinite(stats.fP)
    ? `p ${stats.fP<0.0001?"&lt; 0.0001":"= "+stats.fP.toFixed(4)}`
    : "-";

  const names=["Intercept",...model.features];

  $("statsTable").innerHTML=
    `<tr>
      <th>Predictor</th>
      <th>Estimate (β)</th>
      <th>Std. Error</th>
      <th>t-stat</th>
      <th>p-value</th>
      <th>95% CI</th>
      <th>VIF</th>
    </tr>`+
    names.map((name,i)=>{
      const pVal=stats.pValues[i];
      const pClass=Number.isFinite(pVal) && pVal<0.05 ? "p-good" : "p-warn";
      const vifVal=i===0 ? null : stats.vif[model.features[i-1]];
      const vifClass=Number.isFinite(vifVal) && vifVal<5 ? "vif-good" : "vif-warn";
      return `<tr>
        <td>${name}</td>
        <td>${model.beta[i].toFixed(6)}</td>
        <td>${Number.isFinite(stats.se[i])?stats.se[i].toFixed(6):"-"}</td>
        <td>${Number.isFinite(stats.tStats[i])?stats.tStats[i].toFixed(3):"-"}</td>
        <td><span class="${pClass}">${formatP(pVal)}</span></td>
        <td>[${stats.ci[i][0].toFixed(4)}, ${stats.ci[i][1].toFixed(4)}]</td>
        <td>${i===0?"-":`<span class="${vifClass}">${Number.isFinite(vifVal)?vifVal.toFixed(2):"∞"}</span>`}</td>
      </tr>`;
    }).join("");
}

async function loadDataset(){
  const problem=$("problemSelect").value;
  const custom=document.querySelector('input[name=datasetMode]:checked').value==="custom";
  const url=custom ? $("datasetUrl").value.trim() : DATASETS[problem].url;

  if(!url){alert("請輸入 CSV URL");return;}

  $("datasetMeta").textContent="Loading...";

  Papa.parse(url,{
    download:true,header:true,dynamicTyping:true,skipEmptyLines:true,
    complete: res=>{
      rows=res.data;
      setupColumns(problem);
      renderPreview();
      runAnalysis();
    },
    error: err=>{
      $("datasetMeta").textContent="Load failed: "+err.message;
      alert("Dataset 載入失敗。自訂網址必須能直接下載 CSV，且網站需允許 CORS。");
    }
  });
}

function setupColumns(problem){
  const cols=Object.keys(rows[0]||{});
  const targetSelect=$("targetSelect");
  targetSelect.innerHTML=cols.map(c=>`<option value="${c}">${c}</option>`).join("");

  const preferred=DATASETS[problem].target;
  targetColumn=cols.includes(preferred)?preferred:cols[cols.length-1];
  targetSelect.value=targetColumn;

  rebuildExpandedData();
  updateMaxK();
}

function rebuildExpandedData(){
  targetColumn=$("targetSelect").value;
  const prepared=prepareExpandedData(rows,targetColumn);
  expandedRows=prepared.data;
  expandedFeatures=prepared.featureNames;
}

function updateMaxK(){
  rebuildExpandedData();
  const limit=Math.min(5,expandedFeatures.length);
  $("maxKSelect").innerHTML=Array.from({length:limit},(_,i)=>
    `<option ${i===limit-1?"selected":""}>${i+1}</option>`
  ).join("");
}

function renderPreview(){
  const cols=Object.keys(rows[0]||{});
  $("datasetMeta").textContent=
    `${rows.length} rows · ${cols.length} original columns · ${expandedFeatures.length} model features after encoding`;

  const head=`<tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>`;
  const body=rows.slice(0,10).map(r=>
    `<tr>${cols.map(c=>`<td>${r[c]??""}</td>`).join("")}</tr>`
  ).join("");

  $("previewTable").innerHTML=head+body;
}

function runAnalysis(){
  targetColumn=$("targetSelect").value;
  rebuildExpandedData();
  renderPreview();

  const maxK=Number($("maxKSelect").value);
  currentSplit=splitData(expandedRows,.2);
  progression=buildProgression(
    currentMethod,
    currentSplit.train,
    currentSplit.test,
    expandedFeatures,
    maxK
  );

  if(!progression.length){
    alert("沒有足夠的 Feature 可以建立模型");
    return;
  }

  const globalBest=progression.reduce((a,b)=>a.rmse<b.rmse?a:b);
  selectedK=globalBest.k;

  $("methodDescription").textContent=METHOD_INFO[currentMethod];
  renderAll(currentSplit.test);
}

function renderAll(test){
  const best=progression.reduce((a,b)=>a.rmse<b.rmse?a:b);

  $("bestK").textContent=`k = ${best.k}`;
  $("bestRmse").textContent=best.rmse.toLocaleString(undefined,{maximumFractionDigits:2});
  $("bestR2").textContent=best.r2.toFixed(4);
  $("bestAdjR2").textContent=Number.isFinite(best.adjR2)?best.adjR2.toFixed(4):"-";
  $("bestMae").textContent=best.mae.toLocaleString(undefined,{maximumFractionDigits:2});

  $("kButtons").innerHTML=progression.map(p=>
    `<button class="k-btn ${p.k===selectedK?"active":""}" data-k="${p.k}">
      k = ${p.k}${p.k===best.k?" ★ Best":""}
    </button>`
  ).join("");

  document.querySelectorAll(".k-btn").forEach(b=>
    b.onclick=()=>{
      selectedK=Number(b.dataset.k);
      renderSelected(test);
    }
  );

  renderCharts();
  renderSelected(test);
}

function renderCharts(){
  const labels=progression.map(p=>p.k);

  if(rmseChart) rmseChart.destroy();
  if(r2Chart) r2Chart.destroy();

  rmseChart=new Chart($("rmseChart"),{
    type:"line",
    data:{
      labels,
      datasets:[{
        label:"RMSE",
        data:progression.map(p=>p.rmse),
        tension:.25,
        pointRadius:5
      }]
    },
    options:{
      responsive:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{title:{display:true,text:"Number of Features (k)"}},
        y:{title:{display:true,text:"RMSE"}}
      }
    }
  });

  r2Chart=new Chart($("r2Chart"),{
    type:"line",
    data:{
      labels,
      datasets:[{
        label:"R²",
        data:progression.map(p=>p.r2),
        tension:.25,
        pointRadius:5
      }]
    },
    options:{
      responsive:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{title:{display:true,text:"Number of Features (k)"}},
        y:{title:{display:true,text:"R²"}}
      }
    }
  });
}

function renderSelected(test){
  const p=progression.find(x=>x.k===selectedK)||progression[0];
  currentModel=p.model;

  document.querySelectorAll(".k-btn").forEach(b=>
    b.classList.toggle("active",Number(b.dataset.k)===selectedK)
  );

  $("activeFeatures").innerHTML=
    `<b>Active Predictors (${p.features.length}):</b> `+
    p.features.map(f=>`<span class="feature-pill">${f}</span>`).join("");

  const eq=[
    `${targetColumn} = ${p.model.beta[0].toFixed(4)}`,
    ...p.features.map((f,i)=>
      `${p.model.beta[i+1]>=0?"+":"-"} ${Math.abs(p.model.beta[i+1]).toFixed(4)} × ${f}`
    )
  ].join(" ");

  $("equation").textContent=eq;

  $("featureTableWrap").innerHTML=
    `<table>
      <tr><th>Predictor</th><th>Coefficient</th></tr>
      <tr><td>Intercept</td><td>${p.model.beta[0].toFixed(6)}</td></tr>
      ${p.features.map((f,i)=>
        `<tr><td>${f}</td><td>${p.model.beta[i+1].toFixed(6)}</td></tr>`
      ).join("")}
    </table>`;

  renderStatistics(p.model);
  renderPredictor(p);
  renderResults(p,test);
}

function renderPredictor(p){
  $("predictorFields").innerHTML=p.features.map(f=>{
    const vals=expandedRows.map(r=>Number(r[f]));
    const avg=mean(vals), min=Math.min(...vals), max=Math.max(...vals);

    if(min===0 && max===1 && vals.every(v=>v===0||v===1)){
      return `<div class="field">
        <label>${f}</label>
        <select data-feature="${f}">
          <option value="0" selected>0</option>
          <option value="1">1</option>
        </select>
      </div>`;
    }

    return `<div class="field">
      <label>${f}</label>
      <input type="number" step="any" data-feature="${f}"
        value="${avg.toFixed(2)}" min="${min}" max="${max}">
    </div>`;
  }).join("");
}

function renderResults(p,test){
  const pred=predict(p.model,test);
  const y=test.map(r=>Number(r[targetColumn]));

  const rec=y.map((v,i)=>({
    actual:v,
    pred:pred[i],
    res:v-pred[i],
    abs:Math.abs(v-pred[i]),
    pct:v!==0?Math.abs(v-pred[i])/Math.abs(v)*100:0
  })).sort((a,b)=>b.abs-a.abs);

  $("resultsTable").innerHTML=
    `<tr>
      <th>#</th><th>Actual</th><th>Predicted</th>
      <th>Residual</th><th>Abs Error</th><th>% Error</th>
    </tr>`+
    rec.map((r,i)=>
      `<tr>
        <td>${i+1}</td>
        <td>${r.actual.toFixed(2)}</td>
        <td>${r.pred.toFixed(2)}</td>
        <td>${r.res.toFixed(2)}</td>
        <td>${r.abs.toFixed(2)}</td>
        <td>${r.pct.toFixed(2)}%</td>
      </tr>`
    ).join("");
}

function compareAllMethods(){
  if(!currentSplit) return;

  const maxK=Number($("maxKSelect").value);
  const bestK=Math.min(
    selectedK || 1,
    maxK,
    expandedFeatures.length
  );

  const methods=["lasso","sfs","rfe","elastic","exhaustive"];
  const names={
    lasso:"Lasso",
    sfs:"Forward SFS",
    rfe:"RFE",
    elastic:"ElasticNet",
    exhaustive:"Exhaustive Search"
  };

  const results=[];

  for(const m of methods){
    const selected=chooseFeatures(
      m,
      currentSplit.train,
      expandedFeatures,
      bestK
    );

    const model=fitOLS(currentSplit.train,selected,targetColumn);
    const ev=evaluateModel(model,currentSplit.test);

    results.push({
      method:names[m],
      features:selected,
      rmse:ev.rmse,
      r2:ev.r2,
      mae:ev.mae
    });
  }

  results.sort((a,b)=>a.rmse-b.rmse);

  $("methodComparisonTable").innerHTML=
    `<tr>
      <th>Method</th>
      <th>Features (k=${bestK})</th>
      <th>RMSE</th>
      <th>R²</th>
      <th>MAE</th>
    </tr>`+
    results.map((r,i)=>
      `<tr>
        <td>${i===0?"★ ":""}${r.method}</td>
        <td>${r.features.join(", ")}</td>
        <td>${r.rmse.toFixed(2)}</td>
        <td>${r.r2.toFixed(4)}</td>
        <td>${r.mae.toFixed(2)}</td>
      </tr>`
    ).join("");

  $("methodComparisonWrap").classList.remove("hidden");
}

$("predictBtn").onclick=()=>{
  if(!currentModel) return;

  const fake={};
  document.querySelectorAll("[data-feature]").forEach(el=>
    fake[el.dataset.feature]=Number(el.value)
  );

  const v=predict(currentModel,[fake])[0];
  $("predictionValue").textContent=
    v.toLocaleString(undefined,{maximumFractionDigits:2});
};

document.querySelectorAll(".method-btn").forEach(btn=>{
  btn.onclick=()=>{
    currentMethod=btn.dataset.method;

    document.querySelectorAll(".method-btn").forEach(b=>
      b.classList.toggle("active",b===btn)
    );

    $("methodComparisonWrap").classList.add("hidden");
    runAnalysis();
  };
});

$("compareMethodsBtn").onclick=compareAllMethods;

$("problemSelect").onchange=()=>loadDataset();

$("targetSelect").onchange=()=>{
  updateMaxK();
  runAnalysis();
};

$("maxKSelect").onchange=()=>runAnalysis();

$("loadBtn").onclick=()=>loadDataset();

document.querySelectorAll('input[name=datasetMode]').forEach(r=>
  r.onchange=()=>{
    $("datasetUrl").classList.toggle(
      "hidden",
      r.value!=="custom" || !r.checked
    );

    if(r.checked && r.value==="custom"){
      $("datasetUrl").value=DATASETS[$("problemSelect").value].url;
    }
  }
);

$("methodDescription").textContent=METHOD_INFO[currentMethod];
loadDataset();
