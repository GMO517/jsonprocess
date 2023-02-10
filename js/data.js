const files = document.getElementById('files');
const clearFile_btn = document.querySelector("#clear-btn");
const showPythonFile = document.querySelector('#py-btn');
const jsonShow = document.getElementById('jsonShow')
// const json_file = document.querySelector('#json-btn');
let json_file = document.getElementById('json-link');
let originData = ""; //用於儲存抓進來的資料
const splitKeyWord = "\r\n";
//切割用keyword 需要調整從這邊弄 
const reg = /^\w[a-zA-z]+\:|.*AJCC.*|.*ajcc.*/mg;
// 正則表達式 選取對象 
// 於開頭的 "任意文字:" (含大小寫) or 任何含AJCC的行
let section = originData.split(reg);
// 先切段落
let keyword = ""; //用於儲存段落關鍵字
let contentArray = ""; //用於儲存段落內容
let jsonObject = { "data": [] }; //最後要輸出出去的json物件
let updateArray = [];
let processedData = "";
let contentShowSave = [];
//json檔案宣告
let outputData = "";
let sentenceSave = [];
let structureSave = [];
files.onchange = function () {
  let file = files.files[0];
  let reader = new FileReader();
  reader.readAsText(file);
  reader.onload = function () {
    originData = reader.result;
    //originData = 原始資料 抓進來的資料 
    dataProcess(); //呼叫資料處理的函式
    keywordShow();
    // contentShow();
    let str_obj = JSON.stringify(jsonObject);
    let json = new Blob([str_obj], {
      type: "application/json;charset = utf-8",
    });
    json_file.href = URL.createObjectURL(json);
    json_file.download = 'abc';
    clearOldData(); //每次下載完都要呼叫清理舊資料的函式
  }
}

clearFile_btn.addEventListener("click", clearContent);
//清除資料
// ------------------- python接口 --------------------- //

// ------------------- python接口 --------------------- //
jsonShow.addEventListener('click', function () {
  $.ajax({
    url: "./data/output.json",/*檔案路徑*/
    method: 'GET',
    success: function (jsonData) {
      // console.log(jsonData);
      showJson(jsonData);
    }
  });
});
// ------------------- 接口結束 --------------------- //


//資料顯示函式
function keywordShow() {
  //創造段落區塊
  //keyword塞回段落中
  for (let i = 0; i < keyword.length; i++) {
    const secDiv = document.createElement('div');
    secDiv.classList.add('section' + i);
    secDiv.classList.add('sectionColor');
    secDiv.innerText = keyword[i];
    document.querySelector('.message-show').appendChild(secDiv);
    for (let j = 0; j < contentShowSave[i].length; j++) {
      let content = document.createElement('div');
      content.innerText = contentShowSave[i][j];
      document.querySelector('.section' + i).appendChild(content);
    }
  }
}

//output資料顯示
function showJson(jsonData) {
  for (let i = 0; i < jsonData.data.length; i++) {
    // console.log(jsonData.data[i].Sentence);
    // console.log(jsonData.data[i].Structure);

    //middle 區塊顯示//
    const sentenceDiv = document.createElement('div');
    sentenceDiv.classList.add('sentenceDiv');
    sentenceDiv.innerText = jsonData.data[i].Sentence;
    document.querySelector('.sentence-show').appendChild(sentenceDiv);

    const structureDiv = document.createElement('div');
    structureDiv.innerText = jsonData.data[i].Structure;
    structureDiv.classList.add('structureDiv');
    document.querySelector('.structure-show').appendChild(structureDiv);
    //middle end//


    //sidebar 區塊顯示//
    const sentenceSideBarDiv = document.createElement('div');
    sentenceSideBarDiv.classList.add('sentenceDiv');
    sentenceSideBarDiv.innerText = jsonData.data[i].Sentence;
    document.querySelector('.sentence-sidebar').appendChild(sentenceSideBarDiv);

    const structureSideBarDiv = document.createElement('div');
    structureSideBarDiv.innerText = jsonData.data[i].Structure;
    structureSideBarDiv.classList.add('structureDiv');
    document.querySelector('.structure-sidebar').appendChild(structureSideBarDiv);
    //sidebar end//
  }
}
// //資料處理函式
function dataProcess() {
  keyword = originData.match(reg); //先抓出段落關鍵字
  section = originData.split(reg); //再把各段落切出來
  section = section.filter(v => v); //去除空值
  for (let i = 0; i < section.length; i++) {
    contentArray = section[i].split(splitKeyWord);
    for (let j = 0; j < contentArray.length; j++) {
      if (contentArray[j] === " ") { //去除切割完陣列的空白值
        contentArray[j] = ""; //轉換成空值 之後一起去除
      }
    }
    contentArray = contentArray.filter(v => v); //去除空值
    contentShowSave[i] = contentArray;
    let objContent = {
      "Type": keyword[i],
      "Sentences":
        contentArray
    };
    updateArray.push(objContent); //先push到陣列裡
    jsonObject.data = updateArray; //再把整個陣列丟進object裡
  }
  // console.log(JSON.stringify(jsonObject));
  return jsonObject;
}

function clearOldData() {
  keyword = "";
  section = "";
  contentArray = "";
  updateArray = [];
}
function clearContent() {
  let parent_node = document.querySelector(".message-show")
  let child_node = parent_node.lastElementChild;
  files.value = '';
  if (child_node) {
    while (child_node) {
      parent_node.removeChild(child_node);
      child_node = parent_node.lastElementChild;
    }
  }
}