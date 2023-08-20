document.addEventListener("DOMContentLoaded", function () {
  // Kakao JavaScript SDK의 초기화와 로그인 버튼에 이벤트 핸들러를 설정합니다.
  Kakao.init("4887e7736a873f2e1e22abb50be6b966");

  document
    .getElementById("kakao-login-btn")
    .addEventListener("click", function (event) {
      event.preventDefault(); // 추가된 코드
      console.log("Login button clicked");

      Kakao.Auth.login({
        success: function (authObj) {
          // 로그인 성공 시, 백엔드에 토큰을 전송합니다.
          console.log("Login successful");
          console.log(authObj);
          fetch("http://127.0.0.1:5000/login", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              access_token: authObj.access_token,
            }),
          })
            .then((response) => {
              if (!response.ok) {
                throw new Error(`POST request failed: ${response.status}`);
              }
              return response.json();
            })
            .then((data) => {
              console.log(data);
              if (data.message === "existing_user") {
                alert("이미 계정이 있습니다. 로그인되었습니다.");
              } else if (data.message === "new_user") {
                alert("새로운 계정으로 저장되었습니다.");
              } else {
                alert("알 수 없는 오류가 발생했습니다.");
              }

              localStorage.setItem("jwt", data.token);

              // 로그인 성공 시 index.html로 이동
              window.location.href = "index.html";
            })
            .catch((error) => {
              console.error("Error:", error);
            });
        },
        fail: function (err) {
          console.log(err);
        },
      });
    });
});
