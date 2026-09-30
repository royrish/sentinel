import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fraud Sentinel | Command Center",
  description: "An explainable UPI fraud investigation platform.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}