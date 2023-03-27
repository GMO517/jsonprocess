<?php
  #獲取文件名稱及擴展名稱，將txt擴展名改成json擴展名
    $filename = $_FILES['file']['name'];
    $extension = pathinfo($filename,  PATHINFO_EXTENSION);
    $basename = basename($filename,'.'.$extension);
    $jsonfile = $basename . '.json';
# 檢查檔案是否上傳成功
if ($_FILES['file']['error'] === UPLOAD_ERR_OK){
  echo '檔案名稱: ' . $_FILES['file']['name'] . '<br/>';
  echo '檔案類型: ' . $_FILES['file']['type'] . '<br/>';
  echo '檔案大小: ' . ($_FILES['file']['size'] / 1024) . ' KB<br/>';
  echo '暫存名稱: ' . $_FILES['file']['tmp_name'] . '<br/>';
   
  # 檢查檔案是否已經存在
  if (file_exists('jsonprocess/data/' . $jsonfile)){
    echo '檔案已存在。<br/>';
  } else {
    $file = $_FILES['file']['tmp_name'];
  
    # 將檔案移至指定位置
    $dest = 'jsonprocess/data/' .$jsonfile;
    move_uploaded_file( $file, $dest);
  }
} else {
  echo '錯誤代碼：' . $_FILES['file']['error'] . '<br/>';
}
  
?>