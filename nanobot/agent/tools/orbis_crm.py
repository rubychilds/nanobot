"""Orbis CRM tool: search and manage contacts, notes, and companies.

Credentials (SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_USER_JWT, ORBIS_ORG_ID)
are injected per-request via extra_env and arrive in kwargs["env"].
The LLM never sees these values.
"""

from typing import Any

import httpx

from nanobot.agent.tools.base import Tool

ACTIONS = [
    "search_contacts",
    "get_contact",
    "search_notes",
    "create_note",
    "search_companies",
    "get_company",
    "search_threads",
]


class OrbisCRMTool(Tool):
    """Search and manage the Orbis CRM (contacts, notes, companies, threads)."""

    name = "orbis_crm"
    description = (
        "Search and manage the Orbis CRM. "
        "Actions: search_contacts, get_contact, search_notes, create_note, "
        "search_companies, get_company, search_threads."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ACTIONS,
                "description": "The CRM action to perform.",
            },
            "query": {
                "type": "string",
                "description": "Search query (for search_* actions). Use a person name for contacts, keyword for notes/threads, company name for companies.",
            },
            "contact_id": {
                "type": "string",
                "description": "Contact UUID (for get_contact).",
            },
            "company_id": {
                "type": "string",
                "description": "Company UUID (for get_company).",
            },
            "note_content": {
                "type": "string",
                "description": "Note content in markdown (for create_note).",
            },
            "note_contact_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Contact UUIDs to link to the note (for create_note). At least one required.",
            },
            "note_company_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Company UUIDs to link to the note (for create_note). Optional.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 10, max 25).",
                "minimum": 1,
                "maximum": 25,
            },
        },
        "required": ["action"],
    }

    def _get_credentials(self, env: dict) -> tuple[str, str, str] | None:
        """Extract Supabase credentials from env. Returns None if missing."""
        url = env.get("SUPABASE_URL", "").rstrip("/")
        key = env.get("SUPABASE_ANON_KEY", "")
        jwt = env.get("SUPABASE_USER_JWT", "")
        if not url or not key or not jwt:
            return None
        return url, key, jwt

    def _headers(self, anon_key: str, jwt: str) -> dict[str, str]:
        return {
            "apikey": anon_key,
            "Authorization": f"Bearer {jwt}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=representation",
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        env = kwargs.get("env") or {}
        creds = self._get_credentials(env)
        if creds is None:
            return "Error: Orbis CRM credentials not configured for this session."

        url, anon_key, jwt = creds
        headers = self._headers(anon_key, jwt)
        limit = min(kwargs.get("limit") or 10, 25)

        handlers = {
            "search_contacts": lambda: self._search_contacts(url, headers, kwargs.get("query", ""), limit),
            "get_contact": lambda: self._get_contact(url, headers, kwargs.get("contact_id", "")),
            "search_notes": lambda: self._search_notes(url, headers, kwargs.get("query", ""), limit),
            "create_note": lambda: self._create_note(url, headers, kwargs, env.get("ORBIS_ORG_ID", "")),
            "search_companies": lambda: self._search_companies(url, headers, kwargs.get("query", ""), limit),
            "get_company": lambda: self._get_company(url, headers, kwargs.get("company_id", "")),
            "search_threads": lambda: self._search_threads(url, headers, kwargs.get("query", ""), limit),
        }

        handler = handlers.get(action)
        if not handler:
            return f"Error: Unknown action '{action}'. Valid actions: {', '.join(ACTIONS)}"
        return await handler()

    # ── Search contacts ────────────────────────────────────────────────

    async def _search_contacts(
        self, base_url: str, headers: dict, query: str, limit: int
    ) -> str:
        params: dict[str, Any] = {
            "select": "id,display_name,emails,phones,company,title,location,headline,last_interaction_at,user_tags",
            "limit": limit,
            "order": "last_interaction_at.desc.nullslast",
        }
        if query:
            params["display_name"] = f"ilike.*{query}*"

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{base_url}/rest/v1/user_contact_profiles",
                headers=headers,
                params=params,
            )

        if r.status_code != 200:
            return f"Error searching contacts: {r.status_code} {r.text[:200]}"

        contacts = r.json()
        if not contacts:
            return f"No contacts found for '{query}'." if query else "No contacts found."

        lines = [f"Found {len(contacts)} contact(s):\n"]
        for c in contacts:
            name = c.get("display_name") or "Unknown"
            company = c.get("company") or ""
            title = c.get("title") or ""
            emails = c.get("emails") or []
            tags = ", ".join(c.get("user_tags") or [])
            last = c.get("last_interaction_at", "")[:10] if c.get("last_interaction_at") else "never"

            lines.append(f"- {name} (ID: {c['id']})")
            if title or company:
                role_line = title
                if company:
                    role_line = f"{title} at {company}" if title else company
                lines.append(f"  {role_line}")
            if emails:
                lines.append(f"  Email: {emails[0]}")
            if tags:
                lines.append(f"  Tags: {tags}")
            lines.append(f"  Last interaction: {last}")

        return "\n".join(lines)

    # ── Get contact detail ─────────────────────────────────────────────

    async def _get_contact(self, base_url: str, headers: dict, contact_id: str) -> str:
        if not contact_id:
            return "Error: contact_id is required for get_contact."

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{base_url}/rest/v1/user_contact_profiles",
                headers=headers,
                params={"select": "*", "id": f"eq.{contact_id}", "limit": "1"},
            )

        if r.status_code != 200:
            return f"Error fetching contact: {r.status_code} {r.text[:200]}"

        data = r.json()
        if not data:
            return f"Contact {contact_id} not found."

        c = data[0]
        lines = [f"## {c.get('display_name', 'Unknown')}"]
        for field, label in [
            ("title", "Title"),
            ("company", "Company"),
            ("headline", "Headline"),
            ("location", "Location"),
            ("industry", "Industry"),
        ]:
            if c.get(field):
                lines.append(f"**{label}:** {c[field]}")
        if c.get("emails"):
            lines.append(f"**Emails:** {', '.join(c['emails'])}")
        if c.get("phones"):
            lines.append(f"**Phones:** {', '.join(c['phones'])}")
        if c.get("notes"):
            lines.append(f"**Notes:** {c['notes']}")
        if c.get("summary"):
            lines.append(f"**Summary:** {c['summary']}")
        if c.get("overall_sentiment"):
            lines.append(f"**Sentiment:** {c['overall_sentiment']}")
        if c.get("key_topics"):
            lines.append(f"**Key topics:** {', '.join(c['key_topics'])}")
        if c.get("user_tags"):
            lines.append(f"**Tags:** {', '.join(c['user_tags'])}")
        if c.get("last_interaction_at"):
            lines.append(f"**Last interaction:** {c['last_interaction_at'][:10]}")
        lines.append(f"**ID:** {c['id']}")

        return "\n".join(lines)

    # ── Search notes ───────────────────────────────────────────────────

    async def _search_notes(
        self, base_url: str, headers: dict, query: str, limit: int
    ) -> str:
        params: dict[str, Any] = {
            "select": "id,content,summary,created_at,is_pinned,visibility",
            "order": "created_at.desc",
            "limit": limit,
        }
        if query:
            params["content"] = f"ilike.*{query}*"

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{base_url}/rest/v1/notes",
                headers=headers,
                params=params,
            )

        if r.status_code != 200:
            return f"Error searching notes: {r.status_code} {r.text[:200]}"

        notes = r.json()
        if not notes:
            return "No notes found."

        lines = [f"Found {len(notes)} note(s):\n"]
        for n in notes:
            date = (n.get("created_at") or "")[:10]
            pinned = " [pinned]" if n.get("is_pinned") else ""
            summary = n.get("summary") or ""
            content = n.get("content") or ""
            preview = summary or content[:150].replace("\n", " ")
            if not summary and len(content) > 150:
                preview += "..."
            lines.append(f"- [{date}]{pinned} {preview}")
            lines.append(f"  ID: {n['id']}")

        return "\n".join(lines)

    # ── Create note ────────────────────────────────────────────────────

    async def _create_note(
        self, base_url: str, headers: dict, kwargs: dict, org_id: str
    ) -> str:
        content = (kwargs.get("note_content") or "").strip()
        contact_ids = kwargs.get("note_contact_ids") or []
        company_ids = kwargs.get("note_company_ids") or []

        if not content:
            return "Error: note_content is required for create_note."
        if not contact_ids:
            return "Error: note_contact_ids must include at least one contact UUID."
        if not org_id:
            return "Error: Organization ID not available. Cannot create note."

        body = {
            "p_content": content,
            "p_organization_id": org_id,
            "p_contacts": contact_ids,
            "p_companies": company_ids,
            "p_tags": [],
            "p_visibility": "private",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{base_url}/rest/v1/rpc/create_note",
                headers=headers,
                json=body,
            )

        if r.status_code not in (200, 201):
            return f"Error creating note: {r.status_code} {r.text[:200]}"

        result = r.json()
        note_id = result.get("id", "created") if isinstance(result, dict) else "created"
        return f"Note created successfully. ID: {note_id}"

    # ── Search companies ───────────────────────────────────────────────

    async def _search_companies(
        self, base_url: str, headers: dict, query: str, limit: int
    ) -> str:
        params: dict[str, Any] = {
            "select": "id,name,domain,industry,employee_count,location_name",
            "limit": limit,
            "order": "name.asc",
        }
        if query:
            params["name"] = f"ilike.*{query}*"

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{base_url}/rest/v1/companies",
                headers=headers,
                params=params,
            )

        if r.status_code != 200:
            return f"Error searching companies: {r.status_code} {r.text[:200]}"

        companies = r.json()
        if not companies:
            return f"No companies found for '{query}'." if query else "No companies found."

        lines = [f"Found {len(companies)} company(ies):\n"]
        for co in companies:
            lines.append(f"- {co.get('name', 'Unknown')} (ID: {co['id']})")
            if co.get("domain"):
                lines.append(f"  Domain: {co['domain']}")
            if co.get("industry"):
                lines.append(f"  Industry: {co['industry']}")
            if co.get("employee_count"):
                lines.append(f"  Employees: {co['employee_count']}")
            if co.get("location_name"):
                lines.append(f"  Location: {co['location_name']}")

        return "\n".join(lines)

    # ── Get company detail ─────────────────────────────────────────────

    async def _get_company(self, base_url: str, headers: dict, company_id: str) -> str:
        if not company_id:
            return "Error: company_id is required for get_company."

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{base_url}/rest/v1/companies",
                headers=headers,
                params={"select": "*", "id": f"eq.{company_id}", "limit": "1"},
            )

        if r.status_code != 200:
            return f"Error fetching company: {r.status_code} {r.text[:200]}"

        data = r.json()
        if not data:
            return f"Company {company_id} not found."

        co = data[0]
        lines = [f"## {co.get('name', 'Unknown')}"]
        for field, label in [
            ("domain", "Domain"),
            ("website", "Website"),
            ("industry", "Industry"),
            ("size", "Size"),
            ("employee_count", "Employees"),
            ("founded", "Founded"),
            ("location_name", "Location"),
            ("country", "Country"),
            ("linkedin_url", "LinkedIn"),
            ("funding_stage", "Funding stage"),
            ("revenue_range", "Revenue"),
            ("website_summary", "About"),
        ]:
            if co.get(field):
                lines.append(f"**{label}:** {co[field]}")
        if co.get("tech_stack"):
            lines.append(f"**Tech stack:** {', '.join(co['tech_stack'])}")
        lines.append(f"**ID:** {co['id']}")

        return "\n".join(lines)

    # ── Search threads ─────────────────────────────────────────────────

    async def _search_threads(
        self, base_url: str, headers: dict, query: str, limit: int
    ) -> str:
        params: dict[str, Any] = {
            "select": "id,subject,provider,last_message_at,message_count,labels",
            "order": "last_message_at.desc.nullslast",
            "limit": limit,
        }
        if query:
            params["subject"] = f"ilike.*{query}*"

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{base_url}/rest/v1/threads",
                headers=headers,
                params=params,
            )

        if r.status_code != 200:
            return f"Error searching threads: {r.status_code} {r.text[:200]}"

        threads = r.json()
        if not threads:
            return "No threads found."

        lines = [f"Found {len(threads)} thread(s):\n"]
        for t in threads:
            date = (t.get("last_message_at") or "")[:10]
            subject = t.get("subject") or "(no subject)"
            provider = t.get("provider") or ""
            count = t.get("message_count") or 0
            lines.append(f"- [{date}] [{provider}] {subject} ({count} messages)")
            lines.append(f"  ID: {t['id']}")

        return "\n".join(lines)
