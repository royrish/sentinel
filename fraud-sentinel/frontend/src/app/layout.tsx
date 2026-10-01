import type { Metadata } from "next";
import "./globals.css";
import "./console.css";

export const metadata: Metadata = {
  title: "Fraud Sentinel | Fraud Intelligence",
  description: "Fraud analyst investigation console for UPI transaction alerts.",
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