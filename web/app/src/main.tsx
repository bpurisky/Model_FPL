import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { CorrelationLab } from "./views/CorrelationLab";
import "./design/tokens.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");

createRoot(root).render(
  <StrictMode>
    <CorrelationLab />
  </StrictMode>,
);
