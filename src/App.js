import React, { useEffect, useRef, useState } from 'react';
import { initNotifications, notify } from '@mycv/f8-notification';
import { Howl } from 'howler';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';  // Import WebGL backend
import '@tensorflow/tfjs-backend-cpu';    // Import CPU backend
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import soundURL from './assets/warning-sound.mp3';
import './App.css';

var sound = new Howl({
  src: [soundURL],
  volume: 1.0,
});

const NOT_TOUCH_LABEL = 'not_touch';
const TOUCHED_LABEL = 'touched';
const TRAINING_TIMES = 50;
const TOUCHED_CONFI = 0.8;

function App() {
  const video = useRef(); 
  const mobilenetModule = useRef(); 
  const classifier = useRef(); 
  const [touched, setTouched] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [progress, setProgress] = useState(0); 

  const init = async () => {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log('init...');
    await setUpCamera();
    console.log('setup succesfully');

    mobilenetModule.current = await mobilenet.load();
    classifier.current = knnClassifier.create();
    console.log('setup done');
    console.log('khong cham tay len mat va bam nut train 1');

    setIsModelReady(true);
    initNotifications({ cooldown: 3000 });
  };

  const setUpCamera = () => {
    return new Promise((resolve, reject) => {
      navigator.getUserMedia = navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;

      if (navigator.getUserMedia) {
        navigator.getUserMedia(
          { video: true },
          stream => {
            video.current.srcObject = stream;
            video.current.addEventListener('loadeddata', resolve);
          },
          error => reject(error)
        );
      } else {
        reject();
      }
    });
  };

  const train = async label => {
    if (!isModelReady) {
      console.error('Model chưa sẵn sàng');
      return;
    }

    for (let i = 0; i < TRAINING_TIMES; ++i) {
      // Cập nhật tiến trình huấn luyện mỗi vòng lặp
      setProgress(parseInt((i + 1) / TRAINING_TIMES * 100));
      await training(label);
    }
  };

  const training = label => {
    return new Promise(async resolve => {
      if (!mobilenetModule.current) {
        console.error('mobilenetModule chưa được khởi tạo');
        return;
      }

      const embedding = mobilenetModule.current.infer(
        video.current,
        true
      );
      classifier.current.addExample(embedding, label);
      await sleep(100);
      resolve();
    });
  };

  const run = async () => {
    const embedding = mobilenetModule.current.infer(
      video.current,
      true
    );
    const result = await classifier.current.predictClass(embedding);

    console.log('Label: ', result.label);
    console.log('Confidences: ', result.confidences);

    if (
      result.label === TOUCHED_LABEL &&
      result.confidences[result.label] > TOUCHED_CONFI
    ) {
      console.log('Touched');
      sound.play();
      notify('Bo cay pod ra', { body: 'ban vua hut pod phai khong?' });
      setTouched(true);
    } else {
      console.log('Not Touched');
      setTouched(false);
    }

    await sleep(230);
    run();
  };

  const sleep = (ms = 0) => {
    return new Promise(resolve => setTimeout(resolve, ms));
  };

  useEffect(() => {
    init();
    return () => {};
  }, []);

  return (
    <div className={`main ${touched ? 'touched' : ''}`}>
      <div className='title'>
        Phần mềm hỗ trợ mấy bát nhang vàng
      </div>
      <video 
        ref={video}
        className="video"
        autoPlay 
      />
      <div className="control">
        <button className="btn" onClick={() => train(NOT_TOUCH_LABEL)} disabled={!isModelReady}>Train 1</button>
        <button className="btn" onClick={() => train(TOUCHED_LABEL)} disabled={!isModelReady}>Train 2</button>
        <button className="btn" onClick={() => run()} disabled={!isModelReady}>Run</button>
        <button onClick={() => sound.play()}>Test Sound</button>
      </div>

      {/* Hiển thị tiến trình huấn luyện trên trang web */}
      <div className="progress">
        <p>Tiến trình huấn luyện: {progress}%</p>
      </div>
      <div className="instructions">
        <h3>Hướng dẫn sử dụng</h3>
        <p>
          1. Ấn vào <strong>Train 1</strong> (Tuyệt đối không được cầm pod hút để máy tính nhận diện).<br />
          2. Tiếp theo, ấn vào <strong>Train 2</strong> và hút vài hơi chill đi.<br />
          3. Cuối cùng, ấn vào <strong>Run</strong> để khởi động hệ thống nhắc nhở.<br />
          <em>Note:</em> Có thể ấn vào <strong>Test Sound</strong> để kiểm tra âm thanh nhắc nhở trước.
        </p>
      </div>
    </div>
  );
}

export default App;
