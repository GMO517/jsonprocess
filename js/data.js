const files = document.getElementById('files');
const clearFile_btn = document.querySelector("#clear-btn");
const showPythonFile = document.querySelector('#py-btn');
const jsonShow = document.getElementById('jsonShow')
const sidebarBtn = document.getElementById('sidebar-btn');
// const json_file = document.querySelector('#json-btn');
let json_file = document.getElementById('json-link');
let keyword = ""; //用於儲存段落關鍵字
//json檔案宣告
let outputData = "";
let sentenceSave = [];
let structureSave = [];
let processedData = "";
let contentShowSave = [];

files.onchange = function () {
  let filesArray = Array.from(files.files);
  //獲取fileList,將其轉為陣列
  filesArray.forEach((blob) => {
    let reader = new FileReader();
    reader.readAsText(blob);

    return reader.onload = () => {
      document.querySelector('body').classList.toggle('toggle-sidebar')//控制開合
      const originData = reader.result;
      let fileName = blob.name;
      let jsonObj = dataProcess(originData);
      let str_obj = JSON.stringify(jsonObj);
      let json = new Blob([str_obj], {
        type: "application/json;charset = utf-8",
      });
      // keywordShow();
      // console.log(jsonObj)
      postData(json, fileName);//發送json檔案到後端
      // clearOldData(); //每次下載完都要呼叫清理舊資料的函式
    }

  })

  //非同步發送(POST)資料到後端
  async function postData(json, fileName) {
    //接收到json(blob物件)、fileName(檔案名稱)

    let formData = new FormData();
    //創建表單物件，透過append(key,value,檔案名稱)形式將檔案放入表單
    formData.append('file', json, fileName);

    try {
      const response = await fetch('http://localhost/index.php', {
        method: 'POST',
        body: formData
      }).then(res => res.text()
      ).then(res => {
        console.log(res)
      })


      // console.log(formData.get('file'))
    } catch (error) {
      console.log(error);
    }
  }
}

clearFile_btn.addEventListener("click", clearContent);
//清除資料


//JSON按鈕動態增加(尚未撰寫,暫放)
// json_file.href = URL.createObjectURL(json);
// json_file.download = 'abc';


// ------------------- python接口 --------------------- //
jsonShow.addEventListener('click', function () {

  $.ajax({
    // 這邊要改輸出出來的路徑
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
    sentenceDiv.classList.add('sentenceColor');

    sentenceDiv.innerText = jsonData.data[i].Sentence;
    document.querySelector('.sentence-show').appendChild(sentenceDiv);

    const structureDiv = document.createElement('div');
    structureDiv.innerText = jsonData.data[i].Structure;
    structureDiv.classList.add('structureDiv');
    structureDiv.classList.add('structureColor');
    document.querySelector('.structure-show').appendChild(structureDiv);
    //middle end//


    //sidebar 區塊顯示//
    const sentenceSideBarDiv = document.createElement('div');
    sentenceSideBarDiv.classList.add('sentenceDiv');
    sentenceSideBarDiv.classList.add('sentenceColor');
    sentenceSideBarDiv.innerText = jsonData.data[i].Sentence;
    document.querySelector('.sentence-sidebar').appendChild(sentenceSideBarDiv);

    const structureSideBarDiv = document.createElement('div');
    structureSideBarDiv.classList.add('structureDiv');
    structureSideBarDiv.classList.add('structureColor');
    structureSideBarDiv.innerText = jsonData.data[i].Structure;
    document.querySelector('.structure-sidebar').appendChild(structureSideBarDiv);
    //sidebar end//
  }
}

// //資料處理函式
function dataProcess(originData) {
  const splitKeyWord = "\r\n";
  //切割用keyword 需要調整從這邊弄 
  const reg = /^\w[a-zA-z]+\:|.*AJCC.*|.*ajcc.*/mg;
  // 正則表達式 選取對象 
  // 於開頭的 "任意文字:" (含大小寫) or 任何含AJCC的行
  let section = originData.split(reg);
  // 先切段落
  let contentArray = ""; //用於儲存段落內容
  let jsonObject = { "data": [] }; //最後要輸出出去的json物件
  let updateArray = [];



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
  jsonData = "";
  document.querySelector('body').classList.toggle('toggle-sidebar')

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

  let sentenceParentNode = document.querySelector(".sentence-show");
  let sentenceChild = sentenceParentNode.lastElementChild;
  if (sentenceChild) {
    while (sentenceChild) {
      sentenceParentNode.removeChild(sentenceChild);
      sentenceChild = sentenceParentNode.lastElementChild;
    }
  }

  let structureParentNode = document.querySelector(".structure-show");
  let structureChild = structureParentNode.lastElementChild;
  if (structureChild) {
    while (structureChild) {
      structureParentNode.removeChild(structureChild);
      structureChild = structureParentNode.lastElementChild;
    }
  }

  // modal 清除
  let sentenceSidebarParentNode = document.querySelector(".sentence-sidebar");
  let sentenceSidebarChild = sentenceSidebarParentNode.lastElementChild;
  if (sentenceSidebarChild) {
    while (sentenceSidebarChild) {
      sentenceSidebarParentNode.removeChild(sentenceSidebarChild);
      sentenceSidebarChild = sentenceSidebarParentNode.lastElementChild;
    }
  }

  let structureSidebarParentNode = document.querySelector(".structure-sidebar");
  let structureSidebarChild = structureSidebarParentNode.lastElementChild;
  if (structureSidebarChild) {
    while (structureSidebarChild) {
      structureSidebarParentNode.removeChild(structureSidebarChild);
      structureSidebarChild = structureSidebarParentNode.lastElementChild;
    }
  }
}