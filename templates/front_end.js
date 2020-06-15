"use strict";

let formElem = document.getElementById("formElem");

formElem.onsubmit = async (e) => {
    e.preventDefault();

    await fetch('/post', {
      method: 'POST',
      body: new FormData(formElem)
    });

    let outputImg = document.createElement('img');
    outputImg.src = 'plot.png';
    document.getElementById("result").appendChild(outputImg);
};