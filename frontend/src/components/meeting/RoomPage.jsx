import React from "react";
import { useParams } from "react-router-dom";
import { ZegoUIKitPrebuilt } from "@zegocloud/zego-uikit-prebuilt";
import "./RoomPage.css";
import Navbar from "../Navbar";

const RoomPage = () => {
  const { roomId } = useParams();

  const myMeeting = async (Element) => {
    const appID = Number(import.meta.env.VITE_REACT_APP_ZEGO_APP_ID);
    const serverSecret = import.meta.env.VITE_REACT_APP_ZEGO_SERVER_SECRET;
    const kitToken = ZegoUIKitPrebuilt.generateKitTokenForTest(
      appID,
      serverSecret,
      roomId,
      Date.now().toString(),
      "User"
    );
    const zc = ZegoUIKitPrebuilt.create(kitToken);

    zc.joinRoom({
      container: Element,
      sharedLinks: [
        {
          name: "Copy Link",
          URL: `${window.location.origin}/room/${roomId}`,
        },
      ],
      scenario: {
        mode: ZegoUIKitPrebuilt.OneONoneCall,
      },
    });
  };

  return (
    <>
      <Navbar />
      <div className="room-container">
        <div ref={myMeeting} />
      </div>
    </>
  );
};

export default RoomPage;
