import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ErrorState } from "./ErrorState";
import { LoadingState } from "./LoadingState";

function renderLoading(overrides: Partial<Parameters<typeof LoadingState>[0]> = {}) {
  render(
    <LoadingState
      rootClassName="root"
      title="Explorer"
      file="players.json"
      label="Loading players"
      progress={null}
      {...overrides}
    />,
  );
}

describe("LoadingState", () => {
  it("shows a determinate bar and a KB count when the total is known", () => {
    renderLoading({ progress: { received: 51200, total: 102400 } });
    const bar = screen.getByRole("progressbar", { name: "Loading players" });
    expect(bar).toHaveAttribute("aria-valuenow", "50");
    expect(screen.getByText("players.json — 50 KB of 100 KB")).toBeInTheDocument();
  });

  it("is indeterminate when the total is null", () => {
    renderLoading({ progress: { received: 2048, total: null } });
    const bar = screen.getByRole("progressbar");
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(screen.getByText("players.json — 2 KB")).toBeInTheDocument();
  });

  it("names the file before any bytes arrive, and formats MB on request", () => {
    renderLoading({ file: "panel.parquet", unit: "MB" });
    expect(screen.getByText("requesting panel.parquet")).toBeInTheDocument();
    renderLoading({ file: "panel.parquet", unit: "MB", progress: { received: 1_572_864, total: null } });
    expect(screen.getByText("panel.parquet — 1.5 MB")).toBeInTheDocument();
  });
});

describe("ErrorState", () => {
  it("announces the failure with its message and the export command", () => {
    render(
      <ErrorState
        rootClassName="root"
        title="Explorer"
        heading="players.json did not load."
        error={new Error("HTTP 404")}
        command="players"
      />,
    );
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("players.json did not load.");
    expect(alert).toHaveTextContent("HTTP 404");
    expect(alert).toHaveTextContent("Run python -m web.export players and reload.");
  });

  it("omits the export line when no command is given", () => {
    render(
      <ErrorState
        rootClassName="root"
        title="Form Matrix"
        heading="The panel did not load."
        error={new Error("boom")}
        variant="quiet"
      />,
    );
    expect(screen.getByRole("alert")).not.toHaveTextContent("Run");
  });
});
