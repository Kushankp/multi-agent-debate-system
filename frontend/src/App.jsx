import React, { useState } from "react";
import Header from "./components/Header";
import QueryForm from "./components/QueryForm";
import ConversationLayout from "./components/ConversationLayout";
import "./index.css";

export default function App() {
  const [currentQuery, setCurrentQuery] = useState("");

  return (
    <div className="app-wrapper min-h-screen">
      <Header />
      <QueryForm onSubmit={(q) => setCurrentQuery(q)} />
      <ConversationLayout query={currentQuery} />
    </div>
  );
}
